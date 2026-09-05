import argparse
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from models import MODEL_ID as BASELINE

parser = argparse.ArgumentParser()
parser.add_argument("scheme", choices=["fp8", "nvfp4"], help="quantization scheme to evaluate")
args = parser.parse_args()

QUANT = f"{BASELINE.rstrip('/').split('/')[-1]}-{args.scheme}"

N_SAMPLES = 128  # per bucket
SEQ_LEN = 2048
# Baseline and quant are loaded one at a time (never both in VRAM together -
# needed once the model is big enough that two copies don't fit even on an
# 80GB GPU). KL is truncated to the baseline's top-K tokens per position so
# the cache between the two passes stays tiny instead of holding a full
# vocab-sized distribution; for a peaked LM output distribution this is a
# very close approximation of the exact KL, not the exact value.
TOP_K = 100


def load(name):
    return AutoModelForCausalLM.from_pretrained(
        name, torch_dtype="auto", device_map="auto"
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
        # xlam stores bare function schemas; the chat template expects each
        # tool wrapped OpenAI-style as {"type": "function", "function": {...}}.
        tools = [{"type": "function", "function": t} for t in json.loads(e["tools"])]
        text = tok.apply_chat_template(
            [
                {"role": "user", "content": e["query"]},
                {"role": "assistant", "content": e["answers"]},
            ],
            tools=tools,
            tokenize=False,
        )
        samples.append(("agentic", text))

    return samples


@torch.no_grad()
def main():
    tok = AutoTokenizer.from_pretrained(BASELINE)
    samples = build_samples(tok)

    tokenized = [
        (bucket, tok(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN).input_ids)
        for bucket, text in samples
    ]

    # pass 1: baseline only. Cache its top-K logprobs + token indices per
    # position (topk sorts descending, so index 0 is the argmax / top-1 token).
    ref = load(BASELINE)
    cached = []
    for bucket, ids in tokenized:
        lp = F.log_softmax(ref(ids.cuda()).logits[0, :-1].float(), dim=-1)
        topk_lp, topk_idx = lp.topk(TOP_K, dim=-1)
        cached.append((bucket, ids, topk_lp.cpu(), topk_idx.cpu()))
    del ref
    torch.cuda.empty_cache()

    # pass 2: quant only. Gather its logprobs at the baseline's top-K indices.
    qmodel = load(QUANT)

    stats = defaultdict(lambda: [0.0, 0, 0])  # kl_sum, tokens, top1_agree

    for bucket, ids, ref_topk_lp, ref_topk_idx in cached:
        ref_topk_lp = ref_topk_lp.cuda()
        ref_topk_idx = ref_topk_idx.cuda()

        q_lp = F.log_softmax(qmodel(ids.cuda()).logits[0, :-1].float(), dim=-1)
        q_topk_lp = q_lp.gather(-1, ref_topk_idx)

        # KL(ref || quant) truncated to ref's top-K mass per position
        p = ref_topk_lp.exp()
        kl = (p * (ref_topk_lp - q_topk_lp)).sum(-1)

        s = stats[bucket]
        s[0] += kl.sum().item()
        s[1] += kl.numel()
        # top-1 agreement uses quant's true (full-vocab) argmax, not truncated
        s[2] += (ref_topk_idx[:, 0] == q_lp.argmax(-1)).sum().item()

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
