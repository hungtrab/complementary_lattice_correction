# Working in this repository

## What this is

Reference implementation of Complementary Lattice Correction (CLC), a
post-training-quantization correction that flips quantized weights to adjacent
integer levels to remove the first-moment output shift. See `README.md` for the
user-facing description and the paper for the analysis.

## Conventions that matter

**Sign convention.** The paper's `e = W_q - W`, everywhere. The original
smart-flip code used `W - W_q`; the two cancel out in the final result but every
intermediate differs. If you port code from `smart-flip/` or `FPRAG/`, convert it.

**Layout.** The paper writes `W in R^{d x C}` (input dims x output channels).
PyTorch stores `[out, in]`. Code uses the PyTorch layout with `j` for the row
(output channel) and `i` for the column (input dim). AWQ/GPTQ checkpoint formats
index `[K, N] = [in, out]`, so exporting transposes. The `compressed-tensors`
WNA16 format keeps the packed weight in `[out, in]` order and stores
`weight_shape` explicitly; its signed codes are the affine codes shifted by
`2 ** (bits - 1)`.

**Two weights, two roles.** `LatticeState.float_weights` is what the mean shift
is measured against; `LatticeState.pre_round` is the rounding state that defines
the admissible flip direction. For RTN and AWQ they agree. For GPTQ (App. G.2)
and AdaRound (App. G.4) they must not be conflated.

**Scale precision.** Steps are stored at float16 because that is what every
packed format keeps, and rounded *up* so the nearest-level bound `|e| <= s/2`
that Lemma 1(ii) needs holds at every coordinate. Do not "simplify" this to a
plain cast — rounding down lets the extreme weight in a group clamp and break the
bound, and using float32 makes the simulated model differ from the deployed one
by about one float16 ULP per weight.

**Compacted states.** After correction, `LatticeState.compact()` releases
`float_weights` and `pre_round` and narrows the codes to int8/int16 on CPU.
`residual()`, `normalized_residual()`, `flip_direction()` and `apply_flips()`
then raise rather than silently working against a released tensor. If you need
them after a pipeline run, pass `compact_states=False` — do not re-derive the
weight from the lattice, which would be the quantized weight, not the original.

## Testing

`pytest`. Tests are plain functions with pytest fixtures. Assert the property the
code is supposed to guarantee, not an incidental number.

A specific trap: **do not assert quantization quality via end-to-end loss on a
randomly initialised model.** Random weights have no structure for an
activation-aware method to exploit, and the comparison is noise. Assert the
layer-wise objective each method actually optimizes — mean-shift gain for CLC,
reconstruction error for AWQ/GPTQ.

Regression pins (`tests/test_regression_fixture.py`) will move if the lattice
construction changes. That is expected; recompute and re-pin deliberately rather
than loosening the tolerance.

## Adding a base quantizer

Subclass `LayerQuantizer` and return a `QuantizedLayer` carrying the lattice, the
activation mean **in that lattice's coordinate system**, and the pooled variance.
If the coordinate system is transformed, set `channel_scale` so the pipeline and
the exporter know whether the transform still has to be absorbed elsewhere in the
graph. Appendix G is the specification for what each pipeline changes.

## Exportability

A checkpoint can only be written when the stored weight lies on a uniform group
lattice. `PipelineResult.exportable` tracks this and `notes` says why not. Never
"fix" a non-exportable case by re-quantizing the dequantized weight — that
reintroduces the round-trip error the direct packing exists to avoid.

The supported deployment formats are AWQ (4-bit), GPTQ (2/3/4/8-bit), and
compressed-tensors WNA16 (4/8-bit). `export_checkpoint` also needs the loaded
Hugging Face config when the source is a remote model id; the CLI passes
`model.config.to_dict()` so the output remains loadable instead of containing
only a quantization block. A legacy fake-quantized directory can only be
reprojected, via `convert-legacy-awq`; it cannot recover the discarded original
integer lattice exactly.

The clc/deployment.py module is the single compatibility registry for inference
engines. It distinguishes direct checkpoint loading from engine-specific
conversion and writes deployment.json alongside every packed export. Keep this
registry conservative: vLLM, SGLang, TGI, LMDeploy, and Transformers do not
share the same bit-width matrix, while TensorRT-LLM needs an engine build step
and llama.cpp/Ollama need GGUF. The wrappers in scripts/serve/ should mirror
the registry's commands and must not silently re-quantize a CLC checkpoint.
