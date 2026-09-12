"""Zero-shot downstream evaluation through lm-evaluation-harness."""

from __future__ import annotations

from typing import Dict, List, Optional

PAPER_TASKS = ("arc_challenge", "arc_easy", "boolq", "piqa", "rte")
"""The five tasks whose unweighted average the paper reports."""


def evaluate_tasks(
    model,
    tokenizer,
    tasks: Optional[List[str]] = None,
    batch_size: str | int = "auto",
    num_fewshot: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Run lm-eval on an in-memory model and return its per-task results."""
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "lm-eval is not installed; install the 'eval' extra to run downstream tasks"
        ) from error

    wrapped = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=1 if batch_size == "auto" else batch_size,
    )
    results = lm_eval.simple_evaluate(
        model=wrapped,
        tasks=list(tasks or PAPER_TASKS),
        num_fewshot=num_fewshot,
    )
    return results["results"]


def average_accuracy(results: Dict[str, Dict[str, float]]) -> float:
    """Unweighted mean of per-task accuracy, as reported in the paper's tables."""
    accuracies = [
        metrics[key]
        for metrics in results.values()
        for key in ("acc,none", "acc")
        if key in metrics
    ]
    if not accuracies:
        return 0.0
    return sum(accuracies) / len(accuracies)
