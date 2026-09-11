#!/usr/bin/env python3
"""Summarize Psi-0 SONIC evaluations using the HumanoidArena Table S7 protocol."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


TASKS = {
    "football": "Football",
    "double_desk": "DoubleDesk",
    "pp_box": "P&PBox",
    "opendoor": "OpenDoor",
    "sit_sofa": "SitSofa",
    "boxing": "Boxing",
    "vision_navi": "VisNavi",
}
MODES = ("base", "visual", "semantic", "execution")
HOI = ("football", "double_desk", "pp_box")
HSI = ("opendoor", "sit_sofa", "boxing", "vision_navi")
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_REPEATS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "eval_results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <results-root>/psi0_sonic_table_s7.",
    )
    return parser.parse_args()


def load_complete_run(results_root: Path, task: str, mode: str) -> tuple[Path, list[dict]]:
    candidates = sorted(
        results_root.glob(f"psi0_sonic_{task}_100000_{mode}_*/summary.jsonl"),
        reverse=True,
    )
    for summary_path in candidates:
        rows = [
            json.loads(line)
            for line in summary_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        keys = {(int(row["seed"]), int(row["repeat_idx"])) for row in rows}
        expected = {(seed, repeat) for seed in EXPECTED_SEEDS for repeat in range(EXPECTED_REPEATS)}
        if len(rows) == len(expected) and keys == expected:
            invalid = [
                row
                for row in rows
                if row.get("failure_reason") in {"process_error", "worker_error"}
                or int(row.get("episode_steps", 0)) <= 0
            ]
            if invalid:
                raise RuntimeError(f"Invalid episodes in {summary_path}: {len(invalid)}")
            return summary_path.parent, rows
    raise FileNotFoundError(f"No complete 60-episode result for task={task}, mode={mode}")


def task_metrics(rows: list[dict]) -> dict:
    per_seed: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        per_seed[int(row["seed"])].append(row)
    seed_rates = []
    for seed in EXPECTED_SEEDS:
        seed_rows = per_seed[seed]
        if len(seed_rows) != EXPECTED_REPEATS:
            raise RuntimeError(f"Seed {seed} has {len(seed_rows)} episodes, expected {EXPECTED_REPEATS}")
        seed_rates.append(sum(bool(row["success"]) for row in seed_rows) / EXPECTED_REPEATS)
    reasons = Counter(str(row.get("failure_reason", "")) for row in rows)
    successes = sum(bool(row["success"]) for row in rows)
    falls = reasons["fall"]
    return {
        "episodes": len(rows),
        "successes": successes,
        "falls": falls,
        "success_rate": successes / len(rows),
        "fall_rate": falls / len(rows),
        "seed_success_rates": seed_rates,
        "seed_success_rate_mean": statistics.mean(seed_rates),
        "seed_success_rate_std": statistics.pstdev(seed_rates),
        "failure_reason_counts": dict(sorted(reasons.items())),
    }


def fmt_sr(metric: dict) -> str:
    return f"{metric['seed_success_rate_mean'] * 100:.1f}±{metric['seed_success_rate_std'] * 100:.1f}%"


def suite_metrics(task_data: dict[str, dict], members: tuple[str, ...]) -> dict:
    rates = [rate for task in members for rate in task_data[task]["seed_success_rates"]]
    return {
        "seed_task_success_rates": rates,
        "success_rate_mean": statistics.mean(rates),
        "success_rate_std": statistics.pstdev(rates),
    }


def fmt_suite(metric: dict) -> str:
    return f"{metric['success_rate_mean'] * 100:.2f}±{metric['success_rate_std'] * 100:.2f}%"


def build_report(results_root: Path) -> dict:
    report: dict = {
        "protocol": {
            "model": "Psi-0",
            "gmt": "SONIC",
            "tasks": len(TASKS),
            "modes": list(MODES),
            "seeds": list(EXPECTED_SEEDS),
            "repeats_per_seed": EXPECTED_REPEATS,
            "episodes_per_task_mode": len(EXPECTED_SEEDS) * EXPECTED_REPEATS,
            "total_episodes": len(TASKS) * len(MODES) * len(EXPECTED_SEEDS) * EXPECTED_REPEATS,
        },
        "modes": {},
    }
    for mode in MODES:
        task_data = {}
        source_dirs = {}
        all_rows = []
        for task in TASKS:
            run_dir, rows = load_complete_run(results_root, task, mode)
            source_dirs[task] = str(run_dir)
            task_data[task] = task_metrics(rows)
            all_rows.extend(rows)
        total_falls = sum(row.get("failure_reason") == "fall" for row in all_rows)
        report["modes"][mode] = {
            "episodes": len(all_rows),
            "falls": total_falls,
            "average_fall_rate": total_falls / len(all_rows),
            "tasks": task_data,
            "hoi": suite_metrics(task_data, HOI),
            "hsi": suite_metrics(task_data, HSI),
            "source_dirs": source_dirs,
        }

    base = report["modes"]["base"]
    for mode in MODES:
        mode_data = report["modes"][mode]
        robustness = {}
        for suite in ("hoi", "hsi"):
            base_sr = base[suite]["success_rate_mean"]
            mode_sr = mode_data[suite]["success_rate_mean"]
            robustness[suite] = {
                "absolute_drop_percentage_points": (base_sr - mode_sr) * 100,
                "relative_drop_percent": ((base_sr - mode_sr) / base_sr * 100) if base_sr else None,
                "retention_percent": (mode_sr / base_sr * 100) if base_sr else None,
            }
        mode_data["robustness_vs_base"] = robustness
    return report


def write_markdown(report: dict, path: Path) -> None:
    columns = [
        "Mode", "Method", "AFR↓", "Football", "DoubleDesk", "P&PBox", "HOI AVG",
        "OpenDoor", "SitSofa", "Boxing", "VisNavi", "HSI AVG",
    ]
    lines = [
        "# Psi-0 — HumanoidArena Table S7-style results",
        "",
        "SONIC demonstrations → Psi-0 → SONIC deployment. Each task/mode uses 3 seeds × 20 trials = 60 episodes.",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for mode in MODES:
        data = report["modes"][mode]
        task_data = data["tasks"]
        row = [
            mode.capitalize(),
            "Psi-0",
            f"{data['average_fall_rate'] * 100:.2f}%",
            fmt_sr(task_data["football"]),
            fmt_sr(task_data["double_desk"]),
            fmt_sr(task_data["pp_box"]),
            fmt_suite(data["hoi"]),
            fmt_sr(task_data["opendoor"]),
            fmt_sr(task_data["sit_sofa"]),
            fmt_sr(task_data["boxing"]),
            fmt_sr(task_data["vision_navi"]),
            fmt_suite(data["hsi"]),
        ]
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "## Robustness relative to Base",
        "",
        "| Mode | HOI absolute drop | HOI relative drop | HOI retention | HSI absolute drop | HSI relative drop | HSI retention |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for mode in MODES[1:]:
        robustness = report["modes"][mode]["robustness_vs_base"]
        hoi = robustness["hoi"]
        hsi = robustness["hsi"]
        lines.append(
            f"| {mode.capitalize()} | {hoi['absolute_drop_percentage_points']:.2f} pp | "
            f"{hoi['relative_drop_percent']:.2f}% | {hoi['retention_percent']:.2f}% | "
            f"{hsi['absolute_drop_percentage_points']:.2f} pp | {hsi['relative_drop_percent']:.2f}% | "
            f"{hsi['retention_percent']:.2f}% |"
        )
    lines.extend([
        "",
        "AFR is the fraction of episodes whose `failure_reason` is `fall`, aggregated over all 7 tasks (420 episodes per mode).",
        "Suite AVG mean/std is computed over the task×seed success rates, matching the Table S7 aggregation.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(report: dict, path: Path) -> None:
    fieldnames = [
        "mode", "task", "suite", "episodes", "successes", "falls", "success_rate",
        "seed_success_rate_mean", "seed_success_rate_std", "fall_rate", "seed_0_sr", "seed_1_sr",
        "seed_2_sr", "source_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for mode in MODES:
            mode_data = report["modes"][mode]
            for task in TASKS:
                metric = mode_data["tasks"][task]
                writer.writerow({
                    "mode": mode,
                    "task": task,
                    "suite": "HOI" if task in HOI else "HSI",
                    "episodes": metric["episodes"],
                    "successes": metric["successes"],
                    "falls": metric["falls"],
                    "success_rate": metric["success_rate"],
                    "seed_success_rate_mean": metric["seed_success_rate_mean"],
                    "seed_success_rate_std": metric["seed_success_rate_std"],
                    "fall_rate": metric["fall_rate"],
                    "seed_0_sr": metric["seed_success_rates"][0],
                    "seed_1_sr": metric["seed_success_rates"][1],
                    "seed_2_sr": metric["seed_success_rates"][2],
                    "source_dir": mode_data["source_dirs"][task],
                })


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.results_root / "psi0_sonic_table_s7"
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args.results_root)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(report, output_dir / "task_metrics.csv")
    write_markdown(report, output_dir / "README.md")
    print(f"Wrote Table S7-style Psi-0 report to {output_dir}")


if __name__ == "__main__":
    main()
