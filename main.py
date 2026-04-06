"""Project entrypoint for staged MoE-FFD reproduction."""

from __future__ import annotations

import argparse

from utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoE-FFD reproduction project")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML experiment config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_config(args.config)
    raise NotImplementedError("The end-to-end entrypoint is completed in Step 6.")


if __name__ == "__main__":
    main()
