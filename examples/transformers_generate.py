"""Generate from a CLC checkpoint with Hugging Face Transformers.

The checkpoint's quantization_config selects the corresponding Transformers
backend. Install the backend required by the format before running this
example, for example AutoAWQ for AWQ or GPT-QModel for GPTQ.
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt",
        default="Explain complementary lattice correction in one sentence.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    load_kwargs = {
        "torch_dtype": torch.float16,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.device == "auto" and torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
    elif args.device != "auto":
        load_kwargs["device_map"] = {"": args.device}

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).eval()
    if "device_map" not in load_kwargs:
        model.to("cpu" if args.device == "auto" else args.device)
    input_device = next(model.parameters()).device
    inputs = tokenizer(args.prompt, return_tensors="pt").to(input_device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
