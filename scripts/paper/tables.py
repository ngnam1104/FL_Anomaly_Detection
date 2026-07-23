"""Generate Table II, Table III, and Table IV as CSV and Markdown."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config.settings import table_ii_rows
from scripts.paper.common import METHOD_LABELS, METHOD_ORDER, load_runs


def _markdown(frame: pd.DataFrame) -> str:
    """Render a small GitHub-flavoured table without optional dependencies."""

    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False)]

    def render(values: list[str]) -> str:
        return "| " + " | ".join(value.replace("|", r"\|") for value in values) + " |"

    return "\n".join(
        [render(headers), render(["---"] * len(headers)), *(render(row) for row in rows)]
    )


def _write(frame: pd.DataFrame, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / f"{stem}.csv", index=False)
    (output / f"{stem}.md").write_text(_markdown(frame) + "\n", encoding="utf-8")


def table_ii(output: Path) -> None:
    _write(pd.DataFrame(table_ii_rows()), output, "table_ii_parameters")


def table_iii(results: Path, output: Path) -> None:
    runs, _ = load_runs(results, "scalability")
    rows = []
    for size in sorted(runs["N"].unique()):
        for method in METHOD_ORDER:
            group = runs[(runs["N"] == size) & (runs["method"] == method)]
            if group.empty:
                continue
            rows.append(
                {
                    "N": size,
                    "Method": METHOD_LABELS[method],
                    "Participation": f"{group['participation'].mean():.3f}",
                    "F1 score": (
                        f"{group['f1'].mean():.4f} ± "
                        f"{group['f1'].std(ddof=0):.4f}"
                    ),
                    "Energy (J)": (
                        f"{group['energy_j'].mean():.1f} ± "
                        f"{group['energy_j'].std(ddof=0):.1f}"
                    ),
                }
            )
    _write(pd.DataFrame(rows), output, "table_iii_scalability")


def table_iv(results: Path, output: Path) -> None:
    runs, _ = load_runs(results, "real")
    rows = []
    for method in METHOD_ORDER:
        method_rows = runs[runs["method"] == method]
        if method_rows.empty:
            continue
        row = {"Method": METHOD_LABELS[method]}
        for dataset in ("SMAP", "MSL"):
            group = method_rows[method_rows["dataset"] == dataset]
            row[f"{dataset} PA-F1"] = (
                f"{group['pa_f1'].mean():.4f} ± "
                f"{group['pa_f1'].std(ddof=0):.4f}"
            )
            row[f"{dataset} E (J)"] = (
                f"{group['energy_j'].mean():.1f} ± "
                f"{group['energy_j'].std(ddof=0):.1f}"
            )
        rows.append(row)
    _write(pd.DataFrame(rows), output, "table_iv_real")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/paper"))
    parser.add_argument("--only", choices=("all", "real"), default="all")
    parser.add_argument("--skip-result-tables", action="store_true")
    args = parser.parse_args()
    if args.only == "all":
        table_ii(args.output)
    if not args.skip_result_tables:
        if args.only == "all":
            table_iii(args.results, args.output)
        table_iv(args.results, args.output)


if __name__ == "__main__":
    main()
