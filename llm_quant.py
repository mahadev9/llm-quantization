import json

from datasets import concatenate_datasets, load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

# MODEL_ID = "Qwen/Qwen3.5-0.8B"
# MAX_SEQUENCE_LENGTH = 8192
# NUM_CALIBRATION_SAMPLES = 512

MODEL_ID = "Qwen/Qwen3.5-4B"
MAX_SEQUENCE_LENGTH = 8192
NUM_CALIBRATION_SAMPLES = 512

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype="auto", device_map="cuda:0"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

suffix = "fp8"
recipe = [
    # AWQModifier(),
    QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]),
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
    return {
        "text": tokenizer.apply_chat_template(
            messages, tools=json.loads(e["tools"]), tokenize=False
        )
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


def restripe_prefix(save_dir, old="model.language_model.", new="model."):
    """llmcompressor + transformers 5.14 save decoder weights under
    `model.language_model.*`, but Qwen3_5ForCausalLM reloads them as `model.*`.
    Rewrite the checkpoint keys in place so it loads cleanly (transformers + vLLM).
    """
    import glob
    import os

    from safetensors.torch import load_file, save_file

    rename = lambda k: k.replace(old, new, 1)

    for path in glob.glob(os.path.join(save_dir, "*.safetensors")):
        sd = load_file(path)
        if not any(k.startswith(old) for k in sd):
            continue
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


restripe_prefix(SAVE_DIR)
