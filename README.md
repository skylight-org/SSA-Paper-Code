# SSA-Paper-Code — PQ Importance Sampling (PQIS)

Minimal subset of the sparse-attention-hub research framework, containing
what is needed to run the PQIS benchmark and nothing else. Everything here is
taken from that framework except the package `__init__.py` files (trimmed to
the kept modules) and `scripts/run_pqis_benchmark.py` (new).

This is the PyTorch implementation, run through HuggingFace `transformers`.
The context is prefilled with dense attention; PQIS is applied to the question
tokens and to every generated token.

## The algorithm

`PQImportance` ([`pq_importance.py`](sparse_attention_hub/sparse_attention/research_attention/maskers/fixed/implementations/pq_importance.py))
subclasses `PQCache`, so quantization and scoring are identical: keys past
`init_offset` are product-quantized with k-means, and each query scores every
key against the codebook. What differs is how the scores are turned into a mask.

1. PQ scores are scaled by `scaling` and the additive attention mask, giving
   approximate logits of the true attention distribution. Keys already selected
   by an earlier masker (sink, local) are excluded.
2. The top `heavy_size` keys are kept with mask value `1.0` and removed from the
   sampling pool — this stratum is exactly PQCache.
3. `sample_size` more keys are drawn **with replacement** from
   `softmax(logits / temperature)`. Each drawn key's mask value is its inclusion
   probability `π = 1 − (1 − p)^m`, floored at `1e-4`.

`get_masked_attention_output` divides by the mask value, so the two strata
together form a Horvitz–Thompson estimate of dense attention (weighted
numerator over weighted denominator, as in vAttention) instead of a truncation
of it. That is the point: PQ scores are approximate, and sampling proportional
to them recovers mass that a hard top-k drops.

`sample_size=0` reduces the masker to plain PQCache, which is the baseline to
compare against at equal total budget.

## Layout

```
sparse_attention_hub/
  sparse_attention/
    base.py, metadata.py
    utils/                      Mask, masked-attention kernels, KV helpers
    research_attention/
      base.py                   ResearchAttention: applies maskers in order
      maskers/base.py           ResearchMasker + MaskerRegistry
      maskers/sampling/base.py  kept only for the one-sampling-masker check
      maskers/fixed/
        base.py                 FixedMasker / TopKMasker
        implementations/
          basic_fixed.py        SinkMasker, LocalMasker, CausalMasker
          pq_top_k.py           PQCache (parent of PQImportance)
          pq_importance.py      PQIS
          utils/pq_utils.py     product quantization + batched k-means
  adapters/                     HuggingFace model adapter + model server
  metric_logging/               MicroMetricLogger (density, output error)
benchmark/
  base.py                       Benchmark.run_benchmark orchestration
  benchmark_registry.py         name -> class registry
  longbench/ ruler32k/ mock_benchmark/
scripts/run_pqis_benchmark.py   single-benchmark entry point
tests/                          unit tests for PQImportance (CPU, no model)
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
# install the torch build that matches your CUDA driver first, e.g. CUDA 12:
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

The default PyPI torch wheel targets the newest CUDA release and will not see
the GPU on an older driver; check `python -c "import torch; print(torch.cuda.is_available())"`.
Tested with Python 3.12, torch 2.10.0+cu126 and transformers 5.20.2 on an H200.

A CUDA GPU is required to run the benchmark; the unit tests run on CPU.

## Run one benchmark

```bash
# smoke test: 5 hardcoded samples, no dataset download
python scripts/run_pqis_benchmark.py \
  --benchmark mock_benchmark --subset reading_comprehension \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --max-requests 5 --max-context-length 4096

# RULER-32k
python scripts/run_pqis_benchmark.py \
  --benchmark ruler32k --subset niah_multikey_2 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --heavy-size 0.01 --sample-size 0.02 --max-requests 50

# PQCache baseline at the same total budget
python scripts/run_pqis_benchmark.py \
  --benchmark ruler32k --subset niah_multikey_2 \
  --heavy-size 0.03 --sample-size 0.0
```

The mock contexts are short enough that the masker falls back to full
attention (density 1.0), so the smoke test checks the pipeline, not PQIS.

`--heavy-size` and `--sample-size` are fractions of the key sequence when below
1.0 and absolute key counts otherwise. `--temperature` above 1.0 flattens the
proposal (more exploration, less sensitivity to PQ error); below 1.0 sharpens it
back towards top-k. The PQ knobs (`--pq-bits 6`, `--pq-group-factor 2`,
`--kmeans-iter 10`, `--metric euclidean`) default to PQCache's settings. `--sink-size` is fed to the masker as `init_offset` so
that the PQ codebook skips exactly the keys the sink masker already took.

`--help` lists every flag.

### What a run writes

Into `--result-dir` (default `./results/pqis`):

| file | contents |
| --- | --- |
| `raw_results.csv` | one row per sample with `predicted_answer` |
| `metrics.json` | per-task and overall scores |
| `config.json` | model, masker and generation settings used |
| `micro_metrics.jsonl` | per-layer attention density and output error |

The script prints the overall score plus the averaged density and output error
at the end.

## Calling it from Python

```python
import torch

from benchmark import Ruler32K
from sparse_attention_hub.adapters import ModelAdapterHF
from sparse_attention_hub.sparse_attention.research_attention import ResearchAttentionConfig
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (
    LocalMaskerConfig, PQImportanceConfig, SinkMaskerConfig,
)

config = ResearchAttentionConfig(masker_configs=[
    SinkMaskerConfig(sink_size=128),
    LocalMaskerConfig(window_size=128),
    PQImportanceConfig(
        heavy_size=0.01, sample_size=0.02, temperature=1.0,
        pq_group_factor=2, pq_bits=6, kmeans_iter=10,
        init_offset=128, metric="euclidean",
    ),
])

adapter = ModelAdapterHF(
    model_name="Qwen/Qwen3-4B-Instruct-2507",
    sparse_attention_config=config,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda:0",
)

# process_request prefills the context densely and enables sparse mode
# itself for the question and generated tokens; do not wrap this call in
# adapter.enable_sparse_mode(), which would make the prefill sparse too.
metrics = Ruler32K(["niah_multikey_2"]).run_benchmark(
    adapter, "./results/pqis",
    generation_kwargs={"max_new_tokens": 32},
    request_kwargs={"max_requests": 50, "max_context_length": 32000},
)
```

## Tests

```bash
pytest tests -q
```

CPU only, no model download.
