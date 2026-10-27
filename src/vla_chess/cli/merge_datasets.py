"""Concatenate finished on-disk LeRobot datasets into one, for training on their union.

Runs in the `datagen` pixi env. Example:

    pixi run -e datagen python -m vla_chess.cli.merge_datasets \
        --out data/merged --repo-id vla_chess/merged data/base data/round1
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from vla_chess.dataset.writer import merge_datasets


def main(argv: Sequence[str] | None = None) -> int:
    """Merge the datasets named on the command line.

    Args:
        argv: Command line without the program name; `sys.argv` is read when omitted.

    Returns:
        Process exit status: 0 on success.
    """
    parser = argparse.ArgumentParser(prog="merge-datasets", description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+", help="datasets to merge, in order")
    parser.add_argument("--out", type=Path, required=True, help="where the merged dataset goes")
    parser.add_argument("--repo-id", required=True, help="LeRobot repo id of the merged dataset")
    args = parser.parse_args(argv)
    stats = merge_datasets(args.roots, args.out, args.repo_id)
    print(
        f"merged {len(args.roots)} datasets: {stats.num_episodes} episodes, "
        f"{stats.num_frames} frames in {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
