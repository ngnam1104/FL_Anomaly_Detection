"""Generate every paper figure and result table after all scenarios finish."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MODULES = (
    "scripts.paper.fig4_convergence",
    "scripts.paper.fig5_scalability",
    "scripts.paper.fig6_engineering",
    "scripts.paper.fig7_noniid",
    "scripts.paper.fig8_real",
    "scripts.paper.tables",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/paper"))
    args = parser.parse_args()
    for module in MODULES:
        subprocess.run(
            [
                sys.executable,
                "-m",
                module,
                "--results",
                str(args.results),
                "--output",
                str(args.output),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
