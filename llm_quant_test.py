import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# MODEL_DIR = "Qwen3.5-0.8B-NVFP4"
MODEL_DIR = "Qwen3.5-4B-NVFP4"

# Plain text / coding prompts.
PROMPTS = [
    "Write a Python function that returns the nth Fibonacci number iteratively.",
    "Explain what a race condition is in one paragraph.",
    "Given a list of ints, write a one-liner that returns only the even ones.",
]

# Tool-calling: schema + a query that should trigger a call. Checks the quantized
# model still emits a well-formed tool call (correct name + parsable arguments).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["city"],
            },
        },
    }
]
TOOL_QUERIES = [
    "What's the weather in Tokyo right now, in celsius?",
    "Is it warmer in Paris or Berlin today?",
]


def generate(model, tok, inputs, raw=False):
    inputs = inputs.to("cuda")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=512, do_sample=True, temperature=0.2, top_p=0.9
        )
    gen = out[0][inputs["input_ids"].shape[1] :]
    # raw=True keeps tool-call delimiter tokens visible for inspection
    return (
        tok.decode(gen, skip_special_tokens=not raw).strip(),
        len(gen),
        time.time() - t0,
    )


def main():
    assert torch.cuda.is_available(), "no CUDA device visible"
    print("device:", torch.cuda.get_device_name(0))

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype="auto", device_map="cuda"
    )
    model.eval()
    print(f"[load] {time.time() - t0:.1f}s")

    for prompt in PROMPTS:
        inputs = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        text, n, dt = generate(model, tok, inputs)
        print("=" * 80)
        print("PROMPT:", prompt)
        print(f"({n} tokens, {n / dt:.1f} tok/s)")
        print("-" * 80)
        print(text)

    for query in TOOL_QUERIES:
        inputs = tok.apply_chat_template(
            [{"role": "user", "content": query}],
            tools=TOOLS,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        text, n, dt = generate(model, tok, inputs, raw=True)
        print("=" * 80)
        print("TOOL QUERY:", query)
        print(f"({n} tokens, {n / dt:.1f} tok/s)")
        print("-" * 80)
        print(text)


if __name__ == "__main__":
    main()
