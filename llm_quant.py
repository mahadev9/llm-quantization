import argparse
import glob
import json
import os
import re

from datasets import concatenate_datasets, load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from models import MODEL_ID

SCHEME_MAP = {"fp8": "FP8_DYNAMIC", "nvfp4": "NVFP4"}

parser = argparse.ArgumentParser()
parser.add_argument("scheme", choices=SCHEME_MAP, help="quantization scheme to apply")
args = parser.parse_args()

suffix = args.scheme
scheme = SCHEME_MAP[suffix]

MAX_SEQUENCE_LENGTH = 8192
NUM_CALIBRATION_SAMPLES = 512

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Text-only calibration set never exercises the vision/audio towers, so their
# Linear layers get no activation statistics - exclude them (matters most for
# NVFP4, which needs calibrated input_global_scale; harmless for FP8_DYNAMIC).
MODALITY_IGNORE = [
    "re:.*vision_tower.*",
    "re:.*audio_tower.*",
    "re:.*embed_vision.*",
    "re:.*embed_audio.*",
]

MOE_IGNORE = [
    "re:.*mlp.gate$",
    "re:.*mlp.shared_expert_gate$",
]

# Most quantization-sensitive Linear layers in a transformer block (see
# GPTQ/AWQ literature): down_proj sits right after the MLP nonlinearity and
# sees the widest activation outliers; o_proj is the analogous spot in
# attention. Keeping these full-precision is the cheapest accuracy lever
# available without touching calibration data or requiring AWQ support.
SENSITIVE_IGNORE = [
    "re:.*mlp.down_proj$",
    "re:.*self_attn.o_proj$",
]

# First/last transformer blocks are consistently the most quantization-
# sensitive in the GPTQ/AWQ/INT8 literature - they handle the rawest token
# representations and the final representation before lm_head, so errors
# there don't get absorbed by later layers the way mid-stack errors do.
# Derived from the actual model's module names rather than a config field,
# since layer nesting differs across architectures (flat for Qwen3.5,
# under "language_model." for Gemma4's ForConditionalGeneration wrapper).
layer_indices = {
    int(m.group(1)) for n, _ in model.named_modules() if (m := re.search(r"\.layers\.(\d+)\.", n))
}
FIRST_LAST_IGNORE = [
    f"re:.*layers\\.{min(layer_indices)}\\..*",
    f"re:.*layers\\.{max(layer_indices)}\\..*",
]

recipe = [
    # AWQModifier(),
    QuantizationModifier(
        targets="Linear",
        scheme=scheme,
        ignore=[
            "lm_head",
            *MODALITY_IGNORE,
            *MOE_IGNORE,
            *SENSITIVE_IGNORE,
            *FIRST_LAST_IGNORE,
        ],
    ),
]

# Calibration set shaped like agent/coding traffic: long contexts, raw source
# code, coding instructions, and tool-call / tool-result conversations.
# Field names and configs vary by dataset version - adjust if a load fails.
SEED = 42
N_CODE = int(NUM_CALIBRATION_SAMPLES * 0.35)
N_INSTRUCT = int(NUM_CALIBRATION_SAMPLES * 0.30)
N_TOOLS = NUM_CALIBRATION_SAMPLES - N_CODE - N_INSTRUCT


def sample(dataset, n):
    return dataset.shuffle(seed=SEED).select(range(min(n, len(dataset))))


# 1. Raw code - completion-style, exercises long context and code token stats.
code_ds = sample(
    load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train"),
    N_CODE,
)
code_ds = code_ds.map(
    lambda e: {"text": e["content"]}, remove_columns=code_ds.column_names
)

# 2. Coding instructions - rendered with the chat template.
instruct_ds = sample(
    load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train"), N_INSTRUCT
)
instruct_ds = instruct_ds.map(
    lambda e: {
        "text": tokenizer.apply_chat_template(
            [
                {"role": "user", "content": e["instruction"]},
                {"role": "assistant", "content": e["response"]},
            ],
            tokenize=False,
        )
    },
    remove_columns=instruct_ds.column_names,
)

# 3. Tool calling - chat template WITH the tool schema block inlined, so the
# structured-output / special-token activation patterns are covered.
# Salesforce/xlam-function-calling-60k is gated: `huggingface-cli login` and
# accept the terms on the dataset page first.
tools_ds = sample(
    load_dataset("Salesforce/xlam-function-calling-60k", split="train"), N_TOOLS
)


def render_tools(e):
    messages = [
        {"role": "user", "content": e["query"]},
        {"role": "assistant", "content": e["answers"]},
    ]
    # xlam stores bare function schemas; the chat template expects each tool
    # wrapped OpenAI-style as {"type": "function", "function": {...}}.
    tools = [{"type": "function", "function": t} for t in json.loads(e["tools"])]
    return {
        "text": tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
    }


tools_ds = tools_ds.map(render_tools, remove_columns=tools_ds.column_names)

ds = concatenate_datasets([code_ds, instruct_ds, tools_ds]).shuffle(seed=SEED)

# Apply quantization.
oneshot(
    model=model,
    processor=tokenizer,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    pipeline="basic",  # full forward passes on-GPU; no CPU offload / activation cache
)

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + f"-{suffix}"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)


def fix_saved_key_prefixes(model, save_dir):
    """
    Some ForConditionalGeneration-style wrapper architectures don't
    round-trip save/load consistently across transformers versions (seen with
    both Qwen3.5's and Gemma4's text-decoder nesting - in opposite directions).
    The saved keys can disagree with this model's own module names by a
    missing or extra "language_model." segment after "model.". Detect that
    against ground truth (this model's real module names, not a hardcoded
    architecture list) and repair it in place; a correctly-saved checkpoint
    is left untouched.
    """
    module_names = {n for n, _ in model.named_modules() if n}

    def resolves(key):
        # a leaf tensor's owning module is everything but the last segment
        # (weight / weight_scale / weight_packed / bias / ...); checking any
        # shorter prefix is wrong; nearly every arch has a top-level "model"
        # submodule, which would make that check pass unconditionally.
        parent = key.rsplit(".", 1)[0]
        return parent in module_names

    shard_paths = glob.glob(os.path.join(save_dir, "*.safetensors"))
    keys = []
    for path in shard_paths:
        with safe_open(path, framework="pt") as f:
            keys.extend(f.keys())

    if all(resolves(k) for k in keys):
        return  # already consistent, nothing to fix

    def strip(k):
        return k.replace("model.language_model.", "model.", 1)

    def add(k):
        if k.startswith("model.") and not k.startswith("model.language_model."):
            return "model.language_model." + k[len("model.") :]
        return k

    rename = None
    for transform in (strip, add):
        candidate = lambda k, t=transform: k if resolves(k) else t(k)
        if all(resolves(candidate(k)) for k in keys):
            rename = candidate
            break

    if rename is None:
        print(
            f"warning: could not auto-repair key prefixes in {save_dir}; check manually"
        )
        return

    n_changed = sum(1 for k in keys if rename(k) != k)
    for path in shard_paths:
        sd = load_file(path)
        save_file(
            {rename(k): v for k, v in sd.items()}, path, metadata={"format": "pt"}
        )

    index_path = os.path.join(save_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        index["weight_map"] = {rename(k): v for k, v in index["weight_map"].items()}
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    qcfg = config.get("quantization_config", {})
    if "ignore" in qcfg:
        qcfg["ignore"] = [rename(x) for x in qcfg["ignore"]]
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    print(f"fix_saved_key_prefixes: renamed {n_changed} keys in {save_dir}")


fix_saved_key_prefixes(model, SAVE_DIR)
