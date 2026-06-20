"""CLI entry point for reproducible experiment runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.config import load_run_config
from experiments.pipeline import run_pipeline
from experiments.services.results import render_markdown_table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run reciprocal-rec experiment pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to run config JSON",
    )
    parser.add_argument(
        "--policy-analysis",
        action="store_true",
        help="After eval, write policy_tradeoff.json (engagement vs mutual Recall@K)",
    )
    args = parser.parse_args(argv)
    config = load_run_config(args.config)
    result = run_pipeline(config, policy_analysis=args.policy_analysis)
    print(render_markdown_table(result.table))


if __name__ == "__main__":
    main()
