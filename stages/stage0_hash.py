"""Stage 0 – hash seed + gamma materialisation."""

from __future__ import annotations

import argparse
import secrets
from pathlib import Path

from config import HashStageConfig, ModelSpec
from hashing import HashConfig as HashParams, write_hash_config


def run_stage0(config: HashStageConfig) -> Path:
    seed = config.seed if config.seed is not None else secrets.randbits(63)
    params = HashParams(seed=seed, gamma=config.gamma)
    output = config.resolved_output()
    write_hash_config(output, params)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 0 – hash configuration")
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = HashStageConfig(
        teacher=ModelSpec(name="unused"),
        exp_dir=args.exp_dir,
        seed=args.seed,
        gamma=args.gamma,
        output_file=args.output,
    )
    output = run_stage0(config)
    print(f"Wrote hash config to {output}")


if __name__ == "__main__":
    main()
