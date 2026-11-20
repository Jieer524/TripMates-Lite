"""Build deterministic thesis tables and simple research figures."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "research_final"
SOURCES = [
    ROOT / "results" / "rq1_full_benchmark_corrected_static_rule" / "rq1_raw_runs.csv",
    ROOT / "results" / "weather_ablation" / "weather_ablation_raw_90_pairs.csv",
]

METRICS = ["feasible", "achieved_utility", "completed_poi_count", "original_poi_retention", "wes", "total_travel_min", "edit_distance_score", "runtime_seconds"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for source in SOURCES:
        if source.exists():
            frame = pd.read_csv(source)
            frame["source_artifact"] = source.parent.name
            frames.append(frame)
    if not frames:
        raise FileNotFoundError("No official result CSVs were found.")
    all_runs = pd.concat(frames, ignore_index=True, sort=False)
    available = [c for c in METRICS if c in all_runs.columns]
    grouped = all_runs.groupby("planner", dropna=False)
    summary = grouped[available].agg(["count", "mean", "median", "std"]).reset_index()
    summary.columns = ["_".join(str(x) for x in c if x) if isinstance(c, tuple) else c for c in summary.columns]
    summary.to_csv(OUT / "overall_planner_comparison.csv", index=False)
    all_runs.to_csv(OUT / "combined_official_runs.csv", index=False)

    for label, columns in {
        "feasibility_recovery": ["planner", "feasible", "failure_reason"],
        "utility_retention": ["planner", "achieved_utility", "original_poi_retention", "completed_poi_count"],
        "wes_exposure": ["planner", "wes", "total_outdoor_exposure_min", "outdoor_walking_min", "outdoor_visit_min"],
        "stability_edit_distance": ["planner", "changed_activity_count", "edit_distance_score", "locked_prefix_changed"],
        "runtime": ["planner", "runtime_seconds"],
    }.items():
        cols = [c for c in columns if c in all_runs.columns]
        all_runs[cols].to_csv(OUT / f"{label}.csv", index=False)

    try:
        import matplotlib.pyplot as plt
        means = all_runs.groupby("planner", dropna=False)[[c for c in ("wes", "achieved_utility", "edit_distance_score") if c in all_runs]].mean()
        for metric in means.columns:
            ax = means[metric].plot(kind="bar", title=f"Mean {metric}", ylabel=metric, figsize=(8, 4))
            ax.figure.tight_layout()
            ax.figure.savefig(OUT / f"mean_{metric}.png", dpi=160)
            plt.close(ax.figure)
    except ImportError:
        (OUT / "figures_unavailable.txt").write_text("matplotlib is not installed; CSV tables remain available.\n", encoding="utf-8")

    print(f"Wrote final tables and figures to {OUT.resolve()}")


if __name__ == "__main__":
    main()
