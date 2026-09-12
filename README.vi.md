<p align="center">
  <img src="assets/clc-banner.svg" alt="Complementary Lattice Correction tương thích với vLLM, SGLang, TGI, LMDeploy và Hugging Face Transformers" width="100%">
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.x"></a>
  <a href="https://github.com/vllm-project/vllm"><img src="https://img.shields.io/badge/vLLM-compatible-7C3AED?logo=nvidia&logoColor=white" alt="Tương thích vLLM"></a>
  <a href="https://github.com/hungtrab/complementary_lattice_correction/actions"><img src="https://img.shields.io/badge/tests-157%20passing-16A34A" alt="157 tests passing"></a>
</p>

<p align="center">
  <strong>Hiệu chỉnh lượng tử hóa hậu huấn luyện theo đúng paper</strong><br>
  Hiệu chỉnh lattice số nguyên, giữ nguyên đồ thị suy luận và xuất checkpoint cho hệ thống phục vụ LLM hiện đại.
</p>

<p align="center">
  <a href="README.md">English</a>
  &nbsp;•&nbsp;
  <a href="#bắt-đầu-nhanh">Bắt đầu nhanh</a>
  &nbsp;•&nbsp;
  <a href="#triển-khai">Triển khai</a>
  &nbsp;•&nbsp;
  <a href="#tái-lập-thí-nghiệm">Tái lập thí nghiệm</a>
</p>

# Complementary Lattice Correction (CLC)

Đây là implementation sạch và bám sát paper của **Complementary Lattice
Correction for Post-Training Quantization**. CLC là một bước hiệu chỉnh sau
lượng tử hóa: base quantizer trước hết đưa trọng số lên một integer lattice,
sau đó CLC chọn một số lượng nhỏ code để dịch sang mức lân cận nhằm giảm độ
dịch đầu ra do lượng tử hóa gây ra trên tập calibration.

CLC chỉ thay đổi việc gán code số nguyên. Phương pháp không thêm tham số bias,
không thay đổi kiến trúc model và không cần fine-tuning bằng gradient. Code,
scale và zero point sau hiệu chỉnh được đóng gói trực tiếp thành checkpoint
cho các runtime suy luận phổ biến.

> **Trạng thái:** research software. Nên luôn kiểm tra perplexity và độ chính
> xác task trên model cũng như phân phối calibration thực tế của bạn.

## Ý tưởng chính

Với affine quantizer theo group, ma trận trọng số được biểu diễn bởi

$$
W_q = (Q - Z) \odot S,
$$

trong đó $Q$ là integer code, $Z$ là zero point theo group và $S$ là bước của
lattice. Với activation mean $\hat{\mu}$, CLC tối ưu độ dịch đầu ra ở
first-moment:

$$
\left\| (W_q - W) \hat{\mu} \right\|_2^2.
$$

Trên mỗi output row, phương pháp chọn các phép dịch code một mức trong tập
ứng viên $I_j$, với ngân sách $B_j = \lceil p |I_j| \rceil$. Vì $S$ và $Z$
được giữ cố định, model sau hiệu chỉnh vẫn nằm trên đúng quantization grid và
dùng được cùng low-bit kernel.

## Những gì repository cung cấp

- **Notation và sign convention theo paper:** dùng nhất quán
  $e = W_q - W$, đồng thời biểu diễn rõ residual, support, budget và các
  đại lượng trong bound.
- **Algorithm 1 có thể kiểm thử:** candidate support, adjacent-level move,
  residual ranking, knee mask, per-row budget và discrete argmin được tách
  thành các thành phần độc lập.
- **Ước lượng có tính đến variance:** calibration tích lũy streaming moments
  và pooled activation variance cần cho James–Stein mà không phải giữ toàn bộ
  calibration corpus trong RAM.
- **Bốn base quantizer:** RTN, AWQ, GPTQ và AdaRound dùng chung một pipeline.
- **Export để triển khai:** lattice sau hiệu chỉnh được pack trực tiếp thành
  checkpoint AWQ, GPTQ hoặc compressed-tensors WNA16. Exporter kiểm tra code,
  zero point và float16 scale trước khi ghi file.
- **Đánh giá và phân tích:** perplexity trên WikiText-2/C4, task của
  lm-evaluation-harness, theory verifier theo Appendix E.1 và synthetic
  self-test.
- **Di trú checkpoint cũ:** output fake-quantized từ checkout smart-flip cũ
  có thể được reproject rõ ràng thành thư mục AWQ packed mới.

## Cài đặt

PyTorch không bị pin version để có thể khớp với CUDA runtime trên máy đích.

~~~bash
git clone https://github.com/hungtrab/complementary_lattice_correction.git
cd complementary_lattice_correction

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Khi cần CUDA cụ thể, cài PyTorch tương ứng trước.
python -m pip install torch
python -m pip install -e .

# Tùy chọn: evaluation và export integrations.
python -m pip install -e '.[eval,export]'
~~~

Extra <code>export</code> cài compressed-tensors reference package và
Accelerate. vLLM không phải dependency của package nghiên cứu này; hãy cài
vLLM trong serving environment phù hợp với GPU, CUDA và Triton của bạn.

## Bắt đầu nhanh

Ví dụ sau chạy RTN 4-bit kết hợp CLC và ghi ra checkpoint AWQ packed. Có thể
thay model bằng thư mục Hugging Face local hoặc model identifier phù hợp với
môi trường của bạn.

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

Chạy self-test của phần phân tích:

~~~bash
clc verify
~~~

Chạy perplexity và lm-evaluation-harness:

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

Kiểm tra các assumption và bound trên layer thật:

~~~bash
python -m clc.theory.verify \
  --model /path/to/model \
  --layers 16 \
  --n-calib 32 \
  --seqlen 512
~~~

## Triển khai

CLC ghi ra thư mục checkpoint packed thay vì fake-quantized float weights.
Output bao gồm architecture config gốc của Hugging Face, quantization
metadata, tokenizer và generation config nếu các file này có sẵn.

Có thể kiểm tra artifact trước khi serve:

~~~bash
clc engines
clc inspect --checkpoint ./out/qwen-w4-clc --engine sglang
~~~

Hướng dẫn chi tiết theo từng engine nằm trong
[docs/engines.md](docs/engines.md). Repository có sẵn launch wrapper cho
vLLM, SGLang, TGI và LMDeploy; sidecar
<code>deployment.json</code> tự sinh sẽ ghi rõ engine nào load trực tiếp và
engine nào cần conversion.

### vLLM

~~~bash
# AWQ, chỉ hỗ trợ 4-bit
vllm serve ./out/qwen-w4-clc --quantization awq

# GPTQ, bao gồm cả 3-bit
vllm serve ./out/model-w3-gptq --quantization gptq

# compressed-tensors WNA16; quantization_config tự mô tả format
vllm serve ./out/model-w4-ct
~~~

Hoặc dùng vLLM Python API:

~~~python
from vllm import LLM, SamplingParams

llm = LLM(model="./out/qwen-w4-clc", quantization="awq")
result = llm.generate(
    ["Explain complementary lattice correction in one sentence."],
    SamplingParams(max_tokens=64, temperature=0.0),
)
print(result[0].outputs[0].text)
~~~

### Các engine khác

Format packed AWQ/GPTQ cũng dùng được với SGLang, Hugging Face Text Generation
Inference, LMDeploy và Transformers, tùy backend, GPU và bit width mà engine
đó hỗ trợ:

~~~bash
# SGLang
scripts/serve/sglang.sh ./out/qwen-w4-clc --host 0.0.0.0 --port 30000

# Hugging Face TGI
scripts/serve/tgi.sh ./out/qwen-w4-clc --hostname 0.0.0.0 --port 8080

# LMDeploy TurboMind
scripts/serve/lmdeploy.sh ./out/qwen-w4-clc --server-port 23333

# Transformers
python examples/transformers_generate.py --model ./out/qwen-w4-clc
~~~

TensorRT-LLM có conversion/build path cho W4A16 AWQ/GPTQ. llama.cpp và Ollama
dùng GGUF nên hiện được đánh dấu là conversion target, không phải direct
consumer; việc re-quantize sang GGUF sẽ không còn giữ nguyên CLC lattice.

### Ma trận format

| Format | Số bit | Cờ serving thường dùng | Ràng buộc chính |
| --- | ---: | --- | --- |
| <code>awq</code> | 4-bit | <code>--quantization awq</code> | AWQ GEMM layout; input group và output packing phải phù hợp kernel |
| <code>gptq</code> | 2/3/4/8-bit | <code>--quantization gptq</code> | GPTQ v2 metadata và int32 word packing |
| <code>compressed-tensors</code> | 4/8-bit | tự phát hiện | WNA16 packed weights với shape và group metadata tường minh |

Exporter hiện không ghi GGUF, EXL2 hoặc layout của runtime khác. Nếu cần các
format đó, hãy dùng downstream converter hiểu đúng semantics của runtime đích.

## Thiết kế bám sát paper

### Lattice cố định và one-level move

CLC không quantize lại một float tensor sau hiệu chỉnh. Phương pháp cập nhật
integer code đúng một mức, kiểm tra giới hạn code và dequantize bằng group step
và zero point ban đầu. Do đó output sau hiệu chỉnh vẫn thuộc cùng discrete
lattice với base quantizer.

### James–Stein shrinkage

Calibration tích lũy first và second moments theo streaming. James–Stein dùng
pooled sampling variance thay vì proxy chỉ đo spread của các channel means.
Dùng <code>--no-james-stein</code> khi cần ablation với raw mean.

### AWQ scale folding

AWQ tìm channel scale theo activation. Để output có thể export, pipeline fold
inverse channel scale vào module tạo ra activation khi kiến trúc cho phép. Nhờ
đó stored weight vẫn có uniform per-group lattice. Nếu graph transformation
không hỗ trợ, pipeline báo non-exportable thay vì âm thầm quantize lại.

### Bias correction là baseline

<code>bias_correction</code> được giữ để làm baseline đo lường. Nó có thể thêm
output bias cho layer vốn không có bias, nên không thể biểu diễn trong packed
weight-only format và được pipeline đánh dấu là measurement-only.

## Legacy conversion

Code smart-flip cũ lưu fake-quantized floating-point weights và đã bỏ metadata
của integer lattice. Vì vậy không thể khôi phục bit-for-bit. Hãy chuyển đổi
tường minh và lưu ý rằng converter sẽ project weights lên lattice AWQ mới:

~~~bash
clc convert-legacy-awq \
  --source ./old-results/model \
  --output ./out/legacy-awq
~~~

Thêm <code>--strict</code> nếu muốn dừng khi gặp tensor không hỗ trợ, thay vì
giữ các tensor đó dưới dạng float carry-through weights.

## Tái lập thí nghiệm

Các wrapper trong <code>scripts/bash</code> hỗ trợ chạy raw, corrected và sweep
theo budget:

~~~bash
MODEL=/path/to/model DEVICE=cuda DO_EXPORT=1 \
  scripts/bash/rtn/run.sh

MODEL=/path/to/model BITS_VALUES="3 4" \
  scripts/bash/gptq/sweep.sh

MODEL=/path/to/model EXPORT_FORMAT=compressed-tensors DO_EXPORT=1 \
  scripts/bash/awq/run.sh
~~~

Các biến môi trường thường dùng gồm <code>MODEL</code>, <code>DEVICE</code>,
<code>GROUP_SIZE</code>, <code>N_CALIB</code>, <code>CALIB_SEQLEN</code>,
<code>BUDGETS</code>, <code>DO_EXPORT</code> và <code>EXPORT_FORMAT</code>.

## Kiểm thử

~~~bash
pytest -q
python -m compileall -q clc tests
python -m clc.theory.selftest
bash -n scripts/bash/common.sh scripts/bash/*/*.sh
~~~

Test suite bao phủ lattice invariants, correction guarantees trong paper,
legacy equivalence, estimator, objective của từng quantizer, export
round-trip, compressed-tensors interoperability, tied weights và end-to-end
pipeline.

## Cấu trúc repository

~~~text
clc/
  lattice.py       LatticeState và fixed-grid invariants
  correction.py    Algorithm 1 và legacy compatibility
  estimators.py    James–Stein estimator và knee support
  statistics.py    Streaming moments, covariance và reservoir
  pipeline.py      Block-wise quantization và error propagation
  quantizers/      RTN, AWQ, GPTQ và AdaRound
  baselines/       Classical bias correction
  export/          AWQ, GPTQ, WNA16 packing và legacy conversion
  theory/          Appendix E.1 verification và self-test
  eval/            Perplexity và lm-evaluation-harness
scripts/bash/      Run và sweep wrappers để tái lập quantization
scripts/serve/     Launch wrapper cho vLLM, SGLang, TGI và LMDeploy
examples/          Ví dụ generation với Transformers
docs/              Ghi chú interoperability theo engine
tests/             Unit, regression, interoperability và theory tests
~~~

## Quan hệ với các checkout ban đầu

Hai thư mục sibling <code>smart-flip</code> và <code>FPRAG</code> là các
checkout nghiên cứu ban đầu và không bị chỉnh sửa. Repository này là boundary
implementation có thể duy trì: hợp nhất method, làm rõ assumption của paper
và bổ sung đường dẫn trực tiếp từ integer code sau hiệu chỉnh tới checkpoint
sẵn sàng cho inference.
