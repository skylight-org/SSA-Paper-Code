#!/usr/bin/env python3
"""Run a single benchmark with the PQ importance sampling (PQIS) masker.

The attention stack is the standard PQCache one -- Sink + Local + a PQ-scored
heavy stratum -- with PQImportance in place of PQCache. PQImportance keeps the
top ``heavy_size`` keys by PQ score at weight 1.0 and then draws
``sample_size`` extra keys by multinomial sampling of ``softmax(pq_scores /
temperature)``, storing each drawn key's inclusion probability as the mask
value. get_masked_attention_output divides by that value, so the output is a
Horvitz-Thompson estimate of dense attention rather than a truncation of it.

Setting ``sample_size=0`` reduces the run to plain PQCache, which is the
baseline this is meant to be compared against.

Examples:
    # 5-sample smoke test, no dataset download
    python scripts/run_pqis_benchmark.py --benchmark mock_benchmark \
        --subset reading_comprehension --model Qwen/Qwen3-4B-Instruct-2507 \
        --max-requests 5 --max-context-length 4096

    # RULER-32k, the configuration used for the paper numbers
    python scripts/run_pqis_benchmark.py --benchmark ruler32k \
        --subset niah_multikey_2 --model Qwen/Qwen3-4B-Instruct-2507 \
        --heavy-size 0.01 --sample-size 0.02 --max-requests 50

    # PQCache baseline (same budget, no sampling)
    python scripts/run_pqis_benchmark.py --benchmark ruler32k \
        --subset niah_multikey_2 --heavy-size 0.03 --sample-size 0.0
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Run from the repository root so that `benchmark` and `sparse_attention_hub`
# are importable regardless of where the script is invoked from.
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmark import Benchmark, create_benchmark_instance  # noqa: E402
from sparse_attention_hub.adapters import ModelAdapterHF  # noqa: E402
from sparse_attention_hub.metric_logging import MicroMetricLogger  # noqa: E402
from sparse_attention_hub.sparse_attention.research_attention import (  # noqa: E402
    ResearchAttentionConfig,
)
from sparse_attention_hub.sparse_attention.research_attention.maskers.fixed.implementations import (  # noqa: E402
    LocalMaskerConfig,
    PQImportanceConfig,
    SinkMaskerConfig,
)


def build_pqis_config(args: argparse.Namespace) -> ResearchAttentionConfig:
    """Build the Sink + Local + PQImportance masker stack."""
    return ResearchAttentionConfig(
        masker_configs=[
            SinkMaskerConfig(sink_size=args.sink_size),
            LocalMaskerConfig(window_size=args.window_size),
            PQImportanceConfig(
                heavy_size=args.heavy_size,
                sample_size=args.sample_size,
                temperature=args.temperature,
                pq_group_factor=args.pq_group_factor,
                pq_bits=args.pq_bits,
                kmeans_iter=args.kmeans_iter,
                # PQ scores only the keys past the sink, so this must match it
                init_offset=args.sink_size,
                metric=args.metric,
            ),
        ]
    )


def read_micro_metric_averages(
    micro_metrics_path: Path,
) -> Tuple[Optional[float], Optional[float]]:
    """Average the per-layer density and output error logged during the run."""
    if not micro_metrics_path.exists():
        return None, None

    density_values: List[float] = []
    error_values: List[float] = []
    with micro_metrics_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record: Dict[str, Any] = json.loads(line)
            value = record.get("value")
            if value is None:
                continue
            metric_name: str = str(record.get("metric", ""))
            if metric_name == "research_attention_density":
                density_values.append(float(value))
            elif metric_name == "research_attention_output_error":
                error_values.append(float(value))

    avg_density = sum(density_values) / len(density_values) if density_values else None
    avg_error = sum(error_values) / len(error_values) if error_values else None
    return avg_density, avg_error


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run one benchmark with the PQIS masker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    benchmark_group = parser.add_argument_group("benchmark")
    benchmark_group.add_argument(
        "--benchmark",
        default="ruler32k",
        choices=["ruler32k", "longbench", "mock_benchmark"],
        help="Benchmark to run",
    )
    benchmark_group.add_argument(
        "--subset",
        action="append",
        dest="subsets",
        help="Subset to run; repeat for several. Defaults to every subset.",
    )
    benchmark_group.add_argument("--max-requests", type=int, default=10)
    benchmark_group.add_argument("--max-context-length", type=int, default=32000)
    benchmark_group.add_argument("--max-new-tokens", type=int, default=32)
    benchmark_group.add_argument(
        "--result-dir", default="./results/pqis", help="Where to write results"
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    model_group.add_argument("--device", default="cuda:0")
    model_group.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )

    masker_group = parser.add_argument_group("masker")
    masker_group.add_argument("--sink-size", type=int, default=128)
    masker_group.add_argument("--window-size", type=int, default=128)
    masker_group.add_argument(
        "--heavy-size",
        type=float,
        default=0.01,
        help="Top-k stratum, as a fraction of keys (or an int count if >= 1)",
    )
    masker_group.add_argument(
        "--sample-size",
        type=float,
        default=0.02,
        help="Importance-sampled stratum, same units as --heavy-size. 0 = PQCache",
    )
    masker_group.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help=">1 flattens the proposal (more exploration), <1 sharpens it to top-k",
    )
    masker_group.add_argument("--pq-group-factor", type=int, default=2)
    masker_group.add_argument("--pq-bits", type=int, default=6)
    masker_group.add_argument("--kmeans-iter", type=int, default=10)
    masker_group.add_argument(
        "--metric", default="euclidean", choices=["euclidean", "ip"]
    )

    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the multinomial draws"
    )

    args = parser.parse_args()
    # heavy_size / sample_size are Union[float, int]: a whole number means an
    # absolute key count, a fraction means a fraction of the sequence.
    for name in ("heavy_size", "sample_size"):
        value: float = getattr(args, name)
        if value >= 1.0 or value == 0.0:
            setattr(args, name, int(value))
    return args


def main() -> None:
    """Run one benchmark end to end with PQIS attention."""
    args: argparse.Namespace = parse_args()
    os.chdir(REPO_ROOT)
    torch.manual_seed(args.seed)

    sparse_attention_config: ResearchAttentionConfig = build_pqis_config(args)
    print(
        f"PQIS: sink={args.sink_size} local={args.window_size} "
        f"heavy={args.heavy_size} sample={args.sample_size} "
        f"temperature={args.temperature} pq_bits={args.pq_bits}"
    )

    print(f"Loading {args.model} on {args.device} ...")
    adapter: ModelAdapterHF = ModelAdapterHF(
        model_name=args.model,
        sparse_attention_config=sparse_attention_config,
        model_kwargs={"torch_dtype": getattr(torch, args.dtype)},
        device=args.device,
    )

    benchmark: Benchmark = create_benchmark_instance(args.benchmark, args.subsets)

    result_dir: Path = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    metric_logger: MicroMetricLogger = MicroMetricLogger()
    metric_logger.configure_logging(
        log_path=str(result_dir),
        enabled_metrics=[
            "research_attention_density",
            "research_attention_output_error",
        ],
    )
    metric_logger.flush()

    # process_request prefills the context densely and enables sparse mode
    # itself for the question and generated tokens.
    metrics: Dict[str, Any] = benchmark.run_benchmark(
        adapter=adapter,
        result_dir=str(result_dir),
        generation_kwargs={"max_new_tokens": args.max_new_tokens},
        request_kwargs={
            "max_requests": args.max_requests,
            "max_context_length": args.max_context_length,
        },
    )

    avg_density, avg_error = read_micro_metric_averages(
        result_dir / "micro_metrics.jsonl"
    )
    print("\n=== PQIS results ===")
    print(f"  benchmark:       {args.benchmark} {benchmark.subsets_to_run}")
    score = metrics.get("overall_score", metrics.get("accuracy", "n/a"))
    print(f"  overall score:   {score}")
    print(f"  avg density:     {avg_density if avg_density is not None else 'n/a'}")
    print(f"  avg output err:  {avg_error if avg_error is not None else 'n/a'}")
    print(f"  results written: {result_dir}")


if __name__ == "__main__":
    main()
