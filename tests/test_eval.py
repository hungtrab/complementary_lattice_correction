"""Perplexity normalization and the CLI surface."""

import math

import pytest
import torch

from clc.cli import build_quantizer, main
from clc.eval.lm_eval_runner import PAPER_TASKS, average_accuracy
from clc.eval.perplexity import evaluate_perplexity
from clc.quantizers.awq import AWQQuantizer
from clc.quantizers.base import QuantConfig

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=256,
        )
    ).eval()
    model.config.use_cache = False
    return model


def test_perplexity_is_finite_and_near_uniform_for_a_random_model(tiny_model):
    torch.manual_seed(0)
    tokens = torch.randint(0, 128, (1, 512))
    value = evaluate_perplexity(tiny_model, tokens, max_length=128)
    assert math.isfinite(value)
    # An untrained model over 128 symbols cannot do much better than chance.
    assert 50 < value < 400


def test_overlapping_windows_do_not_change_the_normalization(tiny_model):
    """Each token is scored once, so halving the stride must not deflate the score."""
    torch.manual_seed(0)
    tokens = torch.randint(0, 128, (1, 512))

    disjoint = evaluate_perplexity(tiny_model, tokens, max_length=128, stride=128)
    overlapping = evaluate_perplexity(tiny_model, tokens, max_length=128, stride=64)

    # More context can only help, but the two must stay on the same scale --
    # a length-normalization bug shows up here as a large systematic gap.
    assert overlapping <= disjoint * 1.05
    assert overlapping > disjoint * 0.6


def test_perplexity_rejects_a_sequence_shorter_than_one_window(tiny_model):
    with pytest.raises(ValueError, match="no tokens were scored"):
        evaluate_perplexity(tiny_model, torch.randint(0, 128, (1, 1)), max_length=128)


def test_average_accuracy_reads_both_metric_spellings():
    results = {"arc_easy": {"acc,none": 0.8}, "piqa": {"acc": 0.6}}
    assert average_accuracy(results) == pytest.approx(0.7)


def test_average_accuracy_of_nothing_is_zero():
    assert average_accuracy({}) == 0.0


def test_paper_task_list_matches_the_reported_tables():
    assert PAPER_TASKS == ("arc_challenge", "arc_easy", "boolq", "piqa", "rte")


# -- CLI ---------------------------------------------------------------------

def test_build_quantizer_dispatches_by_name():
    assert isinstance(build_quantizer("awq", QuantConfig()), AWQQuantizer)


def test_build_quantizer_rejects_an_unknown_name():
    with pytest.raises(SystemExit, match="unknown quantizer"):
        build_quantizer("nope", QuantConfig())


def test_verify_subcommand_runs_the_selftest(capsys):
    assert main(["verify"]) == 0
    assert "every check holds" in capsys.readouterr().out


def test_quantize_subcommand_requires_a_model():
    with pytest.raises(SystemExit):
        main(["quantize"])
