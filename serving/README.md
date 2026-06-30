# Host-agnostic serving

Provision a trained model on a chosen backend and get back a client handle.
**Provisioning only** — inference clients are reused from `eval.model`
(`OpenAICompatBackend` covers vLLM + Ollama), never re-implemented here.

Tracked in Vikunja project #176 (#912).

## Layout

```
serving/
  base.py        # ServeBackend ABC + ServingHandle (.openai_client())
  vllm.py        # launch an OpenAI-compatible vLLM server (validated end-to-end)
  sagemaker.py   # deploy an LMI v20 endpoint (supersedes infra/sagemaker/deploy_qwen3.py)
  ollama.py      # register a model via a Modelfile + `ollama create`
  tests/         # 26 wiring tests (no boto3/vllm/ollama imported at test time)
```

The training pipeline emits a **LoRA adapter**, which most backends can't serve
directly. Merge it into a full model first with
`model_training/sft/export.py`, then hand the merged dir to a backend.

## Usage

```python
from serving.vllm import VLLMBackend
from eval.model import GenerateConfig

backend = VLLMBackend(port=8000)
handle = backend.deploy("path/to/merged-model", served_model_name="wm-sft")
try:
    client = handle.openai_client()          # eval.model.OpenAICompatBackend
    print(client.chat([{"role": "user", "content": "hi"}], GenerateConfig()))
finally:
    backend.delete(handle)
```

`SageMakerServeBackend` and `OllamaBackend` share the same `deploy()/delete()`
shape; the SageMaker handle is invoked via boto3 (no `base_url`).

## Validated path: local vLLM with the 0.5B smoke model

The full loop (train → `export.py` merge → `VLLMBackend` serve → query → delete)
was validated on an 8GB RTX 5070 with the Qwen2.5-0.5B smoke adapter.

```bash
# in serving/.venv (uv venv --python 3.11 serving/.venv; uv pip install vllm)
VLLM_USE_FLASHINFER_SAMPLER=0 \
serving/.venv/bin/python - <<'PY'
import os; os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from serving.vllm import VLLMBackend
from eval.model import GenerateConfig
h = VLLMBackend(port=8004).deploy(
    os.path.abspath("model_training/sft/outputs/smoke-merged"),
    served_model_name="wm-smoke",
    extra_args=["--gpu-memory-utilization","0.5","--max-model-len","2048",
                "--enforce-eager","--dtype","float16"])
print(h.openai_client().chat([{"role":"user","content":"hi"}], GenerateConfig(max_tokens=16)))
VLLMBackend().delete(h)
PY
```

### `VLLM_USE_FLASHINFER_SAMPLER=0`

vLLM's FlashInfer sampler JIT-compiles a CUDA kernel at runtime and needs the
full CUDA toolkit (`nvcc`). On a box with only the pip CUDA *runtime* (no
toolkit) that fails with `Could not find nvcc`; disabling the FlashInfer sampler
falls back to the native PyTorch sampler. Not needed where the toolkit /
prebuilt FlashInfer is present (e.g. the SageMaker LMI container).

`--enforce-eager` and a low `--gpu-memory-utilization` keep an 8GB laptop GPU
within budget; drop both for a real GPU.

## Status

- **Validated**: vLLM backend, full loop, 0.5B on local GPU.
- **Wiring-tested only**: SageMaker + Ollama backends (mock-and-defer).
- **Follow-up**: retire `infra/sagemaker/deploy_qwen3.py` in favor of
  `serving/sagemaker.py` and update `infra/sagemaker/README.md` + Vikunja #889;
  validate the S3 LMI serving path (see TODO in `sagemaker.py`).
