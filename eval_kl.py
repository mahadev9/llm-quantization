import json
from collections import defaultdict

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

BASELINE = "Qwen/Qwen3.5-4B"
# QUANT = "Qwen3.5-4B-fp8"
QUANT = "Qwen3.5-4B-nvfp4"

# BASELINE = "Qwen/Qwen3.5-0.8B"
# QUANT = "Qwen3.5-0.8B-fp8"

N_SAMPLES = 128  # per bucket
SEQ_LEN = 2048


def load(name):
    return AutoModelForCausalLM.from_pretrained(
        name, torch_dtype="auto", device_map="cuda:0"
    ).eval()


def build_samples(tok):
    """(bucket, text) pairs: plain prose + agentic tool-calling turns."""
    samples = []

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    for t in (t for t in ds["text"] if len(t) > 200):
        samples.append(("prose", t))
        if sum(b == "prose" for b, _ in samples) >= N_SAMPLES:
            break

    # tool-calling: schema + query + tool call, as the model would actually see it
    tools_ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    tools_ds = tools_ds.shuffle(seed=0).select(range(N_SAMPLES))
    for e in tools_ds:
        text = tok.apply_chat_template(
            [
                {"role": "user", "content": e["query"]},
                {"role": "assistant", "content": e["answers"]},
            ],
            tools=json.loads(e["tools"]),
            tokenize=False,
        )
        samples.append(("agentic", text))

    return samples


@torch.no_grad()
def main():
    tok = AutoTokenizer.from_pretrained(BASELINE)
    samples = build_samples(tok)

    ref = load(BASELINE)
    qmodel = load(QUANT)

    stats = defaultdict(lambda: [0.0, 0, 0])  # kl_sum, tokens, top1_agree

    for bucket, text in samples:
        ids = tok(
            text, return_tensors="pt", truncation=True, max_length=SEQ_LEN
        ).input_ids.cuda()

        ref_lp = F.log_softmax(ref(ids).logits[0, :-1].float(), dim=-1)
        q_lp = F.log_softmax(qmodel(ids).logits[0, :-1].float(), dim=-1)

        # KL(ref || quant) per position
        kl = (ref_lp.exp() * (ref_lp - q_lp)).sum(-1)

        s = stats[bucket]
        s[0] += kl.sum().item()
        s[1] += kl.numel()
        s[2] += (ref_lp.argmax(-1) == q_lp.argmax(-1)).sum().item()

    print(f"baseline : {BASELINE}")
    print(f"quant    : {QUANT}\n")
    print(f"{'bucket':<10}{'tokens':>10}{'mean KL':>12}{'top-1 agree':>14}")
    tot = [0.0, 0, 0]
    for bucket, (kl_sum, n, agree) in stats.items():
        print(f"{bucket:<10}{n:>10}{kl_sum / n:>12.4f}{100 * agree / n:>13.2f}%")
        tot = [tot[0] + kl_sum, tot[1] + n, tot[2] + agree]
    print(
        f"{'overall':<10}{tot[1]:>10}{tot[0] / tot[1]:>12.4f}{100 * tot[2] / tot[1]:>13.2f}%"
    )


if __name__ == "__main__":
    main()
