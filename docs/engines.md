# Inference-engine interoperability

CLC uses the packed checkpoint format as its interoperability boundary. The
correction itself is engine-agnostic: it changes the integer code assignments
while preserving the base quantizer's scale, zero point, group size, and model
architecture.

The exporter currently writes:

| Format | Native consumers in this guide | Bit widths |
| --- | --- | --- |
| AWQ GEMM | vLLM, SGLang, TGI, LMDeploy, Transformers | 4 |
| GPTQ v2 | vLLM, SGLang, TGI, LMDeploy, Transformers | 2/3/4/8, with engine-specific limits |
| compressed-tensors WNA16 | vLLM, SGLang, LMDeploy, Transformers | 4/8, with engine-specific limits |
| GGUF (conversion target) | llama.cpp, Ollama, and other GGUF runtimes | F16 or llama.cpp `Q*` types |

The compatibility registry is available from the command line:

~~~bash
clc engines
clc engines --format awq --bits 4
clc inspect --checkpoint ./out/model-w4-clc
clc inspect --checkpoint ./out/model-w4-clc --engine sglang
~~~

An exported directory also contains <code>deployment.json</code>. It is a
sidecar for humans and tooling; inference engines ignore it and continue to
read the standard <code>config.json</code> and <code>model.safetensors</code>.

## vLLM

vLLM directly consumes all three formats produced by the exporter when its
installed backend supports the selected bit width.

~~~bash
scripts/serve/vllm.sh ./out/model-w4-clc --host 0.0.0.0 --port 8000
FORMAT=gptq scripts/serve/vllm.sh ./out/model-w3-gptq --port 8000
FORMAT=compressed-tensors scripts/serve/vllm.sh ./out/model-w4-ct --port 8000
~~~

The explicit equivalent is:

~~~bash
vllm serve ./out/model-w4-clc --quantization awq
vllm serve ./out/model-w3-gptq --quantization gptq
vllm serve ./out/model-w4-ct
~~~

Use the vLLM quantization guide for GPU/kernel-specific restrictions:
<https://docs.vllm.ai/en/latest/features/quantization/>.

## SGLang

SGLang supports offline pre-quantized loading for AWQ, GPTQ, and
compressed-tensors. The launcher uses <code>--model-path</code> and
<code>--quantization</code>:

~~~bash
scripts/serve/sglang.sh ./out/model-w4-clc --host 0.0.0.0 --port 30000
FORMAT=gptq scripts/serve/sglang.sh ./out/model-w3-gptq --tp-size 2
FORMAT=compressed-tensors scripts/serve/sglang.sh ./out/model-w4-ct
~~~

The direct command is:

~~~bash
python -m sglang.launch_server \
  --model-path ./out/model-w4-clc \
  --quantization awq
~~~

SGLang's backend and hardware matrix is the source of truth for supported
architectures and bit widths:
<https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/quantization.md>.

## Hugging Face Text Generation Inference

TGI uses <code>--quantize</code> rather than <code>--quantization</code>. Its
documented pre-quantized paths are AWQ and GPTQ; use 4-bit outputs for the
widest compatibility:

~~~bash
scripts/serve/tgi.sh ./out/model-w4-clc --hostname 0.0.0.0 --port 8080
FORMAT=gptq scripts/serve/tgi.sh ./out/model-w4-gptq --port 8080
~~~

Equivalent launcher command:

~~~bash
text-generation-launcher \
  --model-id ./out/model-w4-clc \
  --quantize awq
~~~

TGI does not appear in the compressed-tensors direct-load path in this
project's registry. Do not pass a compressed-tensors directory to the TGI
wrapper; choose AWQ/GPTQ or a TGI-native conversion instead.

Reference: <https://huggingface.co/docs/text-generation-inference/en/conceptual/quantization>.

## LMDeploy

LMDeploy's TurboMind engine accepts grouped AWQ, GPTQ, and
compressed-tensors model formats. The wrapper makes the model format explicit:

~~~bash
scripts/serve/lmdeploy.sh ./out/model-w4-clc --server-port 23333
FORMAT=gptq scripts/serve/lmdeploy.sh ./out/model-w4-gptq --server-port 23333
FORMAT=compressed-tensors scripts/serve/lmdeploy.sh ./out/model-w4-ct --server-port 23333
~~~

Equivalent command:

~~~bash
lmdeploy serve api_server ./out/model-w4-clc \
  --backend turbomind \
  --model-format awq
~~~

The LMDeploy CLI and engine configuration documentation is at
<https://lmdeploy.readthedocs.io/en/latest/api/cli.html>.

## Hugging Face Transformers

Transformers can load a compatible packed directory through the
<code>quantization_config</code> in <code>config.json</code>. The required
backend is format-dependent:

~~~bash
python -m pip install autoawq
python examples/transformers_generate.py \
  --model ./out/model-w4-clc \
  --prompt "What does CLC preserve?"
~~~

For GPTQ, install GPT-QModel as described in the Transformers documentation:

~~~bash
python -m pip install gptqmodel
python examples/transformers_generate.py --model ./out/model-w4-gptq
~~~

For compressed-tensors:

~~~bash
python -m pip install compressed-tensors
python examples/transformers_generate.py --model ./out/model-w4-ct
~~~

References:

- <https://huggingface.co/docs/transformers/quantization/awq>
- <https://huggingface.co/docs/transformers/main/quantization/gptq>
- <https://huggingface.co/docs/transformers/v4.49.0/en/quantization/compressed_tensors>

## TensorRT-LLM

TensorRT-LLM has W4A16 AWQ and W4A16 GPTQ workflows, but its final artifact
is a TensorRT engine/checkpoint rather than a generic Hugging Face packed
directory. Treat CLC's AWQ/GPTQ output as an input to the
TensorRT-LLM/Model Optimizer conversion and build workflow:

1. Generate and validate a CLC AWQ or GPTQ checkpoint.
2. Run the TensorRT-LLM or Model Optimizer converter for the target model and
   GPU architecture.
3. Build the final TensorRT engine and benchmark it against the CLC source.

This boundary is intentional: a direct exporter without the target engine
version and GPU architecture would be unreliable.

Reference: <https://nvidia.github.io/TensorRT-LLM/features/quantization.html>.

## llama.cpp and Ollama

llama.cpp and Ollama use GGUF. CLC provides an explicit bridge around the
official llama.cpp tools:

~~~bash
# Build llama.cpp once (the converter script and quantizer are both needed).
git clone https://github.com/ggml-org/llama.cpp /opt/llama.cpp
cmake -S /opt/llama.cpp -B /opt/llama.cpp/build
cmake --build /opt/llama.cpp/build --config Release -j

# Install the Python dependencies required by the converter if necessary.
python -m pip install -r /opt/llama.cpp/requirements.txt
~~~

Convert an ordinary Hugging Face safetensors directory:

~~~bash
clc convert-gguf \
  --source /models/Qwen2.5-0.5B-Instruct \
  --output ./out/qwen-Q4_K_M.gguf \
  --llama-cpp /opt/llama.cpp \
  --outtype f16 \
  --quant-type Q4_K_M
~~~

The same command accepts a packed CLC AWQ, GPTQ, or compressed-tensors
directory. CLC decodes its `(q, s, z)` tensors into a temporary standard HF
`model.safetensors` directory, invokes `convert_hf_to_gguf.py`, and removes the
temporary files when the command finishes:

~~~bash
clc convert-gguf \
  --source ./out/model-w4-clc \
  --output ./out/model-w4-clc-Q4_K_M.gguf \
  --llama-cpp /opt/llama.cpp \
  --quant-type Q4_K_M
~~~

Use `--quant-type NONE` (or `F16`, `BF16`, or `F32`) to create a GGUF without
the second `llama-quantize` phase. For a quantized GGUF, the official workflow
is necessarily a dequantize/re-quantize conversion: the result is useful for
llama.cpp/Ollama but is not bit-exact with the source CLC lattice. The command
writes a `<output>.clc.json` sidecar containing the commands and this guarantee.
Keep the original packed directory when exact CLC deployment is required.

Run the result with llama.cpp:

~~~bash
scripts/serve/llama-cli.sh ./out/model-w4-clc-Q4_K_M.gguf \
  -p "Explain complementary lattice correction in one sentence." \
  -n 64
~~~

Or import it into Ollama with a `Modelfile`:

~~~text
FROM /absolute/path/to/model-w4-clc-Q4_K_M.gguf
~~~

~~~bash
ollama create clc-q4 -f Modelfile
ollama run clc-q4
~~~

The converter is maintained in
[`clc/export/gguf.py`](../clc/export/gguf.py). Its CLI also accepts
`--convert-script`, `--quantize-bin`, `--threads`, `--imatrix`,
`--allow-requantize`, `--pure`, and `--leave-output-tensor` for version-specific
llama.cpp workflows.

References: [llama.cpp HF converter](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py),
[llama.cpp quantization guide](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md),
and [Ollama model files](https://docs.ollama.com/modelfile).

## Compatibility policy

The registry uses three statuses:

- <strong>direct</strong>: the engine reads the CLC output directory directly;
- <strong>conversion</strong>: the engine can use the quantization family after
  an engine-specific conversion/build step. For llama.cpp/Ollama, use
  `clc convert-gguf` and expect a downstream re-quantization boundary;
- <strong>unsupported</strong>: the current output is not a native input.

Kernel support changes across engine releases, GPU architectures, and model
families. The sidecar and CLI provide guidance, not a substitute for a
smoke test on the deployment target.
