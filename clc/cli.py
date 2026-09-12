"""Command line entry point: quantize, correct, evaluate, export, verify."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from clc.correction import CorrectionConfig
from clc.pipeline import PipelineConfig, QuantizationPipeline
from clc.quantizers.adaround import AdaRoundQuantizer
from clc.quantizers.awq import AWQQuantizer
from clc.quantizers.base import QuantConfig
from clc.quantizers.gptq import GPTQQuantizer
from clc.quantizers.rtn import RTNQuantizer

QUANTIZERS = {
    "rtn": RTNQuantizer,
    "awq": AWQQuantizer,
    "gptq": GPTQQuantizer,
    "adaround": AdaRoundQuantizer,
}
NEEDS_SAMPLES = ("awq", "adaround")
"""Quantizers whose objective needs activation rows, not just moments."""


def build_quantizer(name: str, config: QuantConfig, args=None):
    if name not in QUANTIZERS:
        raise SystemExit(f"unknown quantizer {name!r}; expected one of {sorted(QUANTIZERS)}")
    if name == "adaround" and args is not None:
        return AdaRoundQuantizer(
            config,
            iterations=args.adaround_iterations,
            learning_rate=args.adaround_learning_rate,
            batch_size=args.adaround_batch_size,
        )
    return QUANTIZERS[name](config)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="HuggingFace id or local path")
    parser.add_argument("--quantizer", default="rtn", choices=sorted(QUANTIZERS))
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--post-correction", default="clc", choices=["clc", "bias_correction", "none"]
    )
    parser.add_argument("--budget", type=float, default=0.05, help="p in Eq. (8)")
    parser.add_argument("--knee-tolerance", type=float, default=0.0)
    parser.add_argument("--no-james-stein", action="store_true")
    parser.add_argument(
        "--candidate-order",
        default="residual",
        choices=["residual", "normalized"],
        help="rank candidates by |e| (paper) or by |e|/s",
    )
    parser.add_argument(
        "--legacy",
        "--legacy-smartflip",
        dest="legacy",
        action="store_true",
        help="reproduce the pre-refactor smart-flip behaviour bit-for-bit",
    )
    parser.add_argument("--calib-dataset", default="c4", choices=["c4", "wikitext2"])
    parser.add_argument("--n-calib", type=int, default=128)
    parser.add_argument("--calib-seqlen", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="./data/cache")
    parser.add_argument(
        "--device",
        default="auto",
        help="device placement: auto, cpu, cuda, or a concrete torch device",
    )
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--adaround-iterations", type=int, default=2000)
    parser.add_argument("--adaround-learning-rate", type=float, default=1e-3)
    parser.add_argument("--adaround-batch-size", type=int, default=32)


def load_model_and_tokenizer(path: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    # ``torch_dtype`` is understood by the minimum supported Transformers
    # version (4.45); newer releases also accept it as a deprecated alias for
    # ``dtype``.  Keeping the compatibility spelling makes the CLI portable
    # across the model environments used by the two source repositories.
    load_kwargs = {"torch_dtype": torch.float16}
    if device == "auto" and torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
    elif device.startswith("cuda") and torch.cuda.is_available():
        load_kwargs["device_map"] = {"": device}
    model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs).eval()
    if "device_map" not in load_kwargs:
        target = "cpu" if device == "auto" else device
        model.to(target)
    model.config.use_cache = False
    return model, tokenizer


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def run_quantize(args) -> int:
    from clc.calibration import load_calibration_data

    torch.manual_seed(args.seed)
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    batches = load_calibration_data(
        args.calib_dataset,
        tokenizer,
        n_samples=args.n_calib,
        seqlen=args.calib_seqlen,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    execution_device = _model_device(model)
    batches = [b.to(execution_device) for b in batches]

    pipeline = QuantizationPipeline(
        build_quantizer(
            args.quantizer,
            QuantConfig(
                bits=args.bits,
                group_size=args.group_size,
                symmetric=args.symmetric,
            ),
            args,
        ),
        PipelineConfig(
            post_correction=args.post_correction,
            track_covariance=args.quantizer == "gptq",
            sample_limit=2048 if args.quantizer in NEEDS_SAMPLES else 0,
        ),
        CorrectionConfig(
            budget_fraction=args.budget,
            knee_tolerance=args.knee_tolerance,
            use_james_stein=not args.no_james_stein,
            candidate_order=args.candidate_order,
            legacy=args.legacy,
        ),
    )
    result = pipeline.run(model, batches)

    print(f"quantized {len(result.states)} layers with {args.quantizer} @ {args.bits}-bit")
    if args.post_correction == "clc":
        gain = sum(s.get("mean_shift_gain", 0.0) for s in result.layer_stats.values())
        flips = sum(int(s.get("flip_count", 0)) for s in result.layer_stats.values())
        print(f"aggregate mean-shift gain G = {gain:.6g} over K = {flips} flips")
    for note in result.notes:
        print(f"note: {note}")

    if args.perplexity:
        from clc.eval.perplexity import evaluate_perplexity, load_evaluation_tokens

        for dataset in args.perplexity:
            tokens = load_evaluation_tokens(dataset, tokenizer)
            value = evaluate_perplexity(model, tokens, max_length=args.calib_seqlen)
            print(f"{dataset} perplexity: {value:.4f}")

    if args.lm_eval is not None:
        from clc.eval.lm_eval_runner import average_accuracy, evaluate_tasks

        batch_size = args.lm_eval_batch_size
        if isinstance(batch_size, str) and batch_size.isdigit():
            batch_size = int(batch_size)
        task_results = evaluate_tasks(
            model,
            tokenizer,
            tasks=args.lm_eval or None,
            batch_size=batch_size,
            num_fewshot=args.lm_eval_num_fewshot,
        )
        print(f"lm-eval average accuracy: {average_accuracy(task_results):.4f}")

    if args.export:
        from clc.export.checkpoint import export_checkpoint

        if not result.exportable:
            raise SystemExit(
                "this configuration cannot be packed:\n  "
                + "\n  ".join(result.notes or ["see the pipeline notes"])
            )
        quantized_weight_names = {f"{module}.weight" for module in result.states}
        extra = {
            name: tensor
            for name, tensor in model.state_dict().items()
            if name not in quantized_weight_names
        }
        output = export_checkpoint(
            result.states,
            args.export,
            fmt=args.export_format,
            source_dir=args.model if Path(args.model).exists() else None,
            extra_tensors=extra,
            model_config=model.config.to_dict(),
        )
        tokenizer.save_pretrained(output)
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.save_pretrained(output)
        print(f"exported {args.export_format} checkpoint to {output}")

    if args.stats_out:
        Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats_out).write_text(json.dumps(result.layer_stats, indent=2))
    return 0


def run_verify(args) -> int:
    from clc.theory.selftest import main as selftest_main

    return selftest_main()


def run_legacy_conversion(args) -> int:
    """CLI adapter for the intentionally lossy legacy-directory conversion."""
    from clc.export.legacy_convert import convert_legacy_awq_checkpoint

    output = convert_legacy_awq_checkpoint(
        args.source,
        args.output,
        group_size=args.group_size,
        verify=not args.no_verify,
        strict=args.strict,
    )
    print(f"converted legacy checkpoint to {output}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="clc", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    quantize = subparsers.add_parser("quantize", help="quantize, correct, optionally export")
    add_common_arguments(quantize)
    quantize.add_argument("--export", help="write a packed checkpoint to this directory")
    quantize.add_argument(
        "--export-format",
        default="awq",
        choices=["awq", "gptq", "compressed-tensors"],
    )
    quantize.add_argument(
        "--perplexity", nargs="*", default=[], choices=["wikitext2", "c4"]
    )
    quantize.add_argument(
        "--lm-eval",
        nargs="*",
        default=None,
        metavar="TASK",
        help="run lm-evaluation-harness; omit task names to use the paper task set",
    )
    quantize.add_argument("--lm-eval-num-fewshot", type=int)
    quantize.add_argument("--lm-eval-batch-size", default="auto")
    quantize.add_argument("--stats-out", help="write per-layer statistics as JSON")
    quantize.set_defaults(func=run_quantize)

    convert = subparsers.add_parser(
        "convert-legacy-awq",
        help="reproject an old fake-quantized CLC directory into packed AWQ",
    )
    convert.add_argument("--source", required=True, help="old local checkpoint directory")
    convert.add_argument("--output", required=True, help="new packed checkpoint directory")
    convert.add_argument("--group-size", type=int, default=128)
    convert.add_argument("--no-verify", action="store_true")
    convert.add_argument(
        "--strict",
        action="store_true",
        help="fail instead of retaining unsupported 2-D tensors as float weights",
    )

    convert.set_defaults(func=run_legacy_conversion)

    verify = subparsers.add_parser("verify", help="check the analysis on synthetic layers")
    verify.set_defaults(func=run_verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
