<p align="center">
  <img src="assets/clc-banner.svg" alt="Complementary Lattice Correction" width="100%">
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.x"></a>
  <a href="https://github.com/vllm-project/vllm"><img src="https://img.shields.io/badge/vLLM-compatible-7C3AED?logo=nvidia&logoColor=white" alt="vLLM compatible"></a>
  <a href="https://github.com/hungtrab/complementary_lattice_correction/actions"><img src="https://img.shields.io/badge/tests-157%20passing-16A34A" alt="157 tests passing"></a>
</p>

<p align="center">
  <strong>Paper-aligned low-bit correction for post-training quantization</strong><br>
  Correct the integer lattice, preserve the inference graph, and export a checkpoint for modern LLM serving.
</p>

<p align="center">
  <a href="README.vi.md">Tiếng Việt</a>
  &nbsp;•&nbsp;
  <a href="#quick-start">Quick start</a>
  &nbsp;•&nbsp;
  <a href="#deployment">Deployment</a>
  &nbsp;•&nbsp;
  <a href="#reproducibility">Reproducibility</a>
</p>

# Complementary Lattice Correction (CLC)

This repository is a clean, paper-oriented reference implementation of
**Complementary Lattice Correction for Post-Training Quantization**. CLC is a
back-end correction stage: a base quantizer first places the weights on an
integer lattice, then CLC selectively moves a small number of codes to an
adjacent level to reduce the calibration-set output shift.

The correction changes integer assignments only. It does not add a bias
parameter, change the model architecture, or require gradient-based
fine-tuning. The resulting codes, scales, and zero points are exported
directly to packed inference checkpoints.

> **Status:** research software. The implementation is designed for controlled
> experiments and reproducible comparisons; always validate perplexity and
> task accuracy for your own model and calibration distribution.

## Why CLC?

For a group-wise affine quantizer, a weight matrix is represented as

$$
W_q = (Q - Z) \odot S,
$$

where $Q$ is the integer code, $Z$ is the group zero point, and $S$ is the
group step. For a calibration mean $\hat{\mu}$, CLC scores the first-moment
output shift

$$
\left\| (W_q - W) \hat{\mu} \right\|_2^2.
$$

For each output row, the method selects admissible one-level moves under a
per-row budget $B_j = \lceil p |I_j| \rceil$. The move keeps $S$ and $Z$
fixed, so the corrected model remains on the same quantization grid and can be
served by the same low-bit kernel.

## What is included?

- **Paper notation and sign convention.** The implementation uses
  $e = W_q - W$ consistently and keeps the paper's residual, support, budget,
  and bound quantities explicit.
- **Algorithm 1.** Candidate support, adjacent-level moves, residual ranking,
  knee-based masking, the per-row budget, and the discrete argmin are separate
  and testable operations.
- **Variance-aware estimation.** Streaming calibration statistics retain the
  pooled activation variance needed by the James–Stein estimator without
  materializing the complete calibration corpus.
- **Multiple base quantizers.** RTN, AWQ, GPTQ, and AdaRound are available
  behind one pipeline.
- **Deployment-ready export.** Corrected lattices are packed directly as AWQ,
  GPTQ, or compressed-tensors WNA16 checkpoints. The exporter verifies codes,
  zero points, and float16 scales before writing.
- **Evaluation and analysis.** WikiText-2/C4 perplexity, lm-evaluation-harness
  tasks, Appendix E.1 theory verification, and synthetic self-tests are
  included.
- **Legacy migration.** Old fake-quantized outputs from the original
  smart-flip checkout can be explicitly reprojected into a packed AWQ
  directory.

## Installation

PyTorch is intentionally left unpinned so that it can match the CUDA runtime
on the target machine.

~~~bash
git clone https://github.com/hungtrab/complementary_lattice_correction.git
cd complementary_lattice_correction

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install PyTorch separately when a specific CUDA build is required.
python -m pip install torch
python -m pip install -e .

# Optional evaluation and export integrations.
python -m pip install -e '.[eval,export]'
~~~

The <code>export</code> extra installs the compressed-tensors reference package
and Accelerate. vLLM is deliberately not a runtime dependency of the research
package; install it in the serving environment that matches your GPU and
CUDA/Triton stack.

## Quick start

The following example runs 4-bit RTN plus CLC and writes a packed AWQ
checkpoint. Replace the model with a local Hugging Face directory or a model
identifier available to your environment.

~~~bash
clc quantize \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --quantizer rtn \
  --bits 4 \
  --group-size 128 \
  --post-correction clc \
  --n-calib 32 \
  --calib-seqlen 512 \
  --device auto \
  --export ./out/qwen-w4-clc \
  --export-format awq
~~~

Run the analysis self-test:

~~~bash
clc verify
~~~

Run perplexity and lm-evaluation-harness after quantization:

~~~bash
clc quantize \
  --model /path/to/model \
  --quantizer awq \
  --bits 4 \
  --post-correction clc \
  --perplexity wikitext2 c4 \
  --lm-eval arc_easy hellaswag piqa \
  --export ./out/model-w4-awq-clc \
  --export-format awq
~~~

For a real-layer analysis of the assumptions and bounds:

~~~bash
python -m clc.theory.verify \
  --model /path/to/model \
  --layers 16 \
  --n-calib 32 \
  --seqlen 512
~~~

## Deployment

CLC writes normal packed checkpoint directories rather than fake-quantized
float weights. The output includes the original Hugging Face architecture
configuration, quantization metadata, tokenizer files, and generation
configuration when available.

### vLLM

~~~bash
# AWQ, 4-bit only
vllm serve ./out/qwen-w4-clc --quantization awq

# GPTQ, including 3-bit
vllm serve ./out/model-w3-gptq --quantization gptq

# compressed-tensors WNA16; quantization_config.json is self-describing
vllm serve ./out/model-w4-ct
~~~

The same directories can be passed to the vLLM Python API:

~~~python
from vllm import LLM, SamplingParams

llm = LLM(model="./out/qwen-w4-clc", quantization="awq")
result = llm.generate(
    ["Explain complementary lattice correction in one sentence."],
    SamplingParams(max_tokens=64, temperature=0.0),
)
print(result[0].outputs[0].text)
~~~

### Format matrix

| Export format | Supported weights | Typical serving flag | Important constraint |
| --- | ---: | --- | --- |
| <code>awq</code> | 4-bit | <code>--quantization awq</code> | AWQ GEMM layout; input groups and output packing must fit the kernel |
| <code>gptq</code> | 2/3/4/8-bit | <code>--quantization gptq</code> | GPTQ v2 metadata and int32 word packing |
| <code>compressed-tensors</code> | 4/8-bit | auto-detected | WNA16 packed weights with explicit shape and group metadata |

The packed exporter is format-specific and intentionally does not emit GGUF,
EXL2, or other unrelated layouts. Convert to those formats only with a
downstream tool that understands the target runtime's semantics.

## Command-line reference

The main command is:

~~~text
clc quantize [OPTIONS]
~~~

Important options:

| Option | Values / default | Purpose |
| --- | --- | --- |
| <code>--quantizer</code> | <code>rtn</code>, <code>awq</code>, <code>gptq</code>, <code>adaround</code> | Select the base PTQ method |
| <code>--bits</code> | 2, 3, 4, 8 | Weight precision |
| <code>--group-size</code> | 128 | Input-channel group size |
| <code>--post-correction</code> | <code>clc</code>, <code>bias_correction</code>, <code>none</code> | Select the post-quantization stage |
| <code>--budget</code> | 0.05 | Fraction of eligible coordinates available to CLC |
| <code>--knee-tolerance</code> | 0.0 | Shift the activation-support knee |
| <code>--no-james-stein</code> | disabled by default | Use the raw activation mean |
| <code>--candidate-order</code> | <code>residual</code> | Use the paper's $|e_{ij}|$ ordering; <code>normalized</code> is available for ablations |
| <code>--legacy</code> | disabled by default | Reproduce the pre-refactor smart-flip behavior |
| <code>--perplexity</code> | optional | Evaluate WikiText-2 and/or C4 |
| <code>--lm-eval</code> | optional task list | Run lm-evaluation-harness |
| <code>--export-format</code> | <code>awq</code> | Select the packed deployment format |

See all flags with:

~~~bash
clc quantize --help
~~~

## Paper-aligned design

### Fixed lattice and one-level moves

CLC never re-quantizes a corrected float tensor. It updates the integer code
by exactly one level, checks the code bounds, and dequantizes with the original
group step and zero point. This makes the corrected output a member of the
same discrete lattice as the base quantizer.

### James–Stein shrinkage

Calibration accumulates first and second moments in a streaming pass. The
James–Stein estimate uses the pooled sampling variance rather than a
spread-of-means proxy. Disable it with <code>--no-james-stein</code> when an
ablation requires the raw mean.

### AWQ scale folding

AWQ searches activation-aware channel scales. For an exportable result, the
pipeline folds the inverse channel scale into the producing module where the
model architecture permits it. This preserves a uniform per-group lattice in
the stored weight. Unsupported graph transformations are reported as
non-exportable instead of being silently re-quantized.

### Bias correction is a baseline

<code>bias_correction</code> is retained as a measurement baseline. It can add
an output bias to a bias-free layer, so it is not represented by the packed
weight-only formats and is marked measurement-only by the pipeline.

## Exportability and legacy conversion

Most standard transformer linear layers are exportable when their input
dimension is divisible by the selected group size. Every layer is checked
before writing:

- codes must remain within the integer range;
- the stored group lattice must not require padding;
- channel-scale transforms must already be folded;
- all exported layers must share bits, group size, and symmetry convention.

The original <code>smart-flip</code> implementation saved fake-quantized
floating-point weights and discarded the original integer lattice metadata.
Those files cannot be recovered bit-for-bit. Convert them explicitly, knowing
that the converter projects them onto a fresh AWQ lattice:

~~~bash
clc convert-legacy-awq \
  --source ./old-results/model \
  --output ./out/legacy-awq
~~~

Use <code>--strict</code> to fail on unsupported tensors instead of retaining
them as float carry-through weights.

## Evaluation

The optional evaluation extra provides lm-evaluation-harness integration:

~~~bash
python -m pip install -e '.[eval]'

clc quantize \
  --model /path/to/model \
  --quantizer rtn \
  --bits 3 \
  --post-correction clc \
  --perplexity wikitext2 c4 \
  --lm-eval arc_easy arc_challenge hellaswag piqa winogrande \
  --lm-eval-num-fewshot 0 \
  --lm-eval-batch-size auto
~~~

Perplexity uses a sliding-window objective with target-token normalization.
The theory verifier instruments real linear layers one at a time so that
covariance collection remains bounded by the selected layer width.

## Reproducibility

The scripts under <code>scripts/bash</code> provide thin wrappers for raw,
corrected, and budget-sweep runs:

~~~bash
MODEL=/path/to/model DEVICE=cuda DO_EXPORT=1 \
  scripts/bash/rtn/run.sh

MODEL=/path/to/model BITS_VALUES="3 4" \
  scripts/bash/gptq/sweep.sh

MODEL=/path/to/model EXPORT_FORMAT=compressed-tensors DO_EXPORT=1 \
  scripts/bash/awq/run.sh
~~~

Common environment variables include <code>MODEL</code>, <code>DEVICE</code>,
<code>GROUP_SIZE</code>, <code>N_CALIB</code>, <code>CALIB_SEQLEN</code>,
<code>BUDGETS</code>, <code>DO_EXPORT</code>, and <code>EXPORT_FORMAT</code>.

## Repository layout

~~~text
clc/
  lattice.py       LatticeState and fixed-grid invariants
  correction.py    Paper Algorithm 1 and legacy compatibility
  estimators.py    James–Stein estimator and knee support
  statistics.py    Streaming moments, covariance, and reservoirs
  pipeline.py      Block-wise quantization and error propagation
  quantizers/      RTN, AWQ, GPTQ, and AdaRound
  baselines/       Classical bias correction
  export/          AWQ, GPTQ, WNA16 packing, and legacy conversion
  theory/          Appendix E.1 verification and self-test
  eval/            Perplexity and lm-evaluation-harness
scripts/bash/      Reproducible run and sweep wrappers
tests/             Unit, regression, interoperability, and theory tests
~~~

## Testing

~~~bash
pytest -q
python -m compileall -q clc tests
python -m clc.theory.selftest
bash -n scripts/bash/common.sh scripts/bash/*/*.sh
~~~

The test suite covers lattice invariants, the paper's correction guarantees,
legacy equivalence, estimator behavior, quantizer objectives, export
round-trips, compressed-tensors interoperability, tied weights, and
end-to-end pipeline behavior.

## Relationship to the original checkouts

The sibling <code>smart-flip</code> and <code>FPRAG</code> directories are
original research checkouts and remain untouched. This repository is the
maintainable implementation boundary: it consolidates the method, makes the
paper assumptions explicit, and adds a direct path from corrected integer
codes to inference-ready checkpoints.

## Citation

If you use CLC in research, please cite the accompanying paper:

> **Complementary Lattice Correction for Post-Training Quantization.**

The final bibliographic entry and paper URL will be added here when the public
version is released.
