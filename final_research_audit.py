"""Audit official research artefacts and validate the common metric schema.

Exit status is non-zero for critical missing/invalid artefacts. Smoke-only
directories are reported separately and are never counted as official runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from benchmark_common import MATRIX_FILE, POI_FILE, file_fingerprint, runtime_metadata

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTPUT = ROOT / "output" / "research_final"
REQUIRED = {
    "planner", "scenario_id", "seed", "feasible", "failure_reason",
    "achieved_utility", "completed_poi_count", "runtime_seconds",
}
STABILITY = {
    "bounded_repair_feasible", "modification_count",
    "itinerary_edit_distance", "locked_prefix_violation",
}
OFFICIAL_DIRS = (
    RESULTS / "rq1_full_benchmark_corrected_static_rule",
    RESULTS / "weather_ablation",
    RESULTS / "thermal_tradeoff_sensitivity_alpha_independent_v2",
    RESULTS / "ablation_studies" / "stability_ablation",
    RESULTS / "ablation_studies" / "full_replanning_ablation",
    RESULTS / "ablation_studies" / "disruption_severity_sensitivity",
    RESULTS / "ablation_studies" / "itinerary_density_sensitivity",
    RESULTS / "ablation_studies" / "weather_profile_sensitivity",
)


def _hashes() -> dict:
    return {"poi_dataset": file_fingerprint(POI_FILE), "walking_matrix": file_fingerprint(MATRIX_FILE)}


def _csv_files(directory: Path) -> list[Path]:
    # Only raw run tables are validated as planner-run records. Summaries,
    # paired-test tables and Pareto fronts intentionally have different
    # schemas and are audited as derived artefacts, not run records.
    return [
        p for p in directory.glob("*.csv")
        if "failure" not in p.name
        and ("raw" in p.name or p.name == "raw_runs.csv")
    ]


def audit(*, allow_missing_official: bool = False) -> dict:
    findings: list[dict] = []
    official: list[dict] = []
    critical = False
    seen: set[tuple] = set()

    for directory in OFFICIAL_DIRS:
        if not directory.exists():
            findings.append({"severity": "critical", "type": "missing_directory", "path": str(directory)})
            critical = True
            continue
        raw_files = _csv_files(directory)
        config_candidates = [directory / "run_config.json"] + list(directory.glob("*_run_config.json"))
        config = next((p for p in config_candidates if p.exists()), directory / "run_config.json")
        if not raw_files:
            findings.append({"severity": "critical", "type": "missing_result_file", "path": str(directory)})
            critical = True
            continue
        if not config.exists():
            findings.append({"severity": "critical", "type": "missing_configuration", "path": str(config)})
            critical = True
        for raw in raw_files:
            try:
                frame = pd.read_csv(raw)
            except Exception as exc:
                findings.append({"severity": "critical", "type": "unreadable_csv", "path": str(raw), "error": str(exc)})
                critical = True
                continue
            missing = sorted(REQUIRED - set(frame.columns))
            missing_stability = sorted(STABILITY - set(frame.columns))
            if missing:
                findings.append({"severity": "critical", "type": "missing_metric_columns", "path": str(raw), "columns": missing})
                critical = True
            if missing_stability:
                findings.append({"severity": "warning", "type": "missing_stability_columns", "path": str(raw), "columns": missing_stability})
            for col in ("runtime_seconds", "achieved_utility", "completed_poi_count"):
                if col in frame and (pd.to_numeric(frame[col], errors="coerce") < 0).any():
                    findings.append({"severity": "critical", "type": "impossible_metric", "path": str(raw), "column": col})
                    critical = True
            key_cols = [c for c in ("experiment_id", "instance_id", "scenario_id", "planner", "seed", "alpha", "selection_policy") if c in frame]
            if len(key_cols) >= 3:
                duplicates = frame.duplicated(key_cols).sum()
                if duplicates:
                    findings.append({"severity": "critical", "type": "duplicate_run_keys", "path": str(raw), "count": int(duplicates), "keys": key_cols})
                    critical = True
            official.append({"path": str(raw), "rows": int(len(frame)), "columns": list(frame.columns)})

    smoke_files = [str(p) for p in RESULTS.rglob("smoke/*.csv")]
    report = {
        "audit_version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "critical_error": critical and not allow_missing_official,
        "allow_missing_official": allow_missing_official,
        "runtime": runtime_metadata(),
        "inputs": _hashes(),
        "official_artifacts": official,
        "smoke_artifacts": smoke_files,
        "findings": findings,
        "summary": {"official_csv_count": len(official), "smoke_csv_count": len(smoke_files), "finding_count": len(findings)},
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "reproducibility_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Reproducibility audit", "", f"Generated: {report['generated_utc']}", f"Critical error: {report['critical_error']}", "", "## Findings"]
    lines += [f"- {f['severity']}: {f['type']} — {f.get('path', '')}" for f in findings] or ["- none"]
    (OUTPUT / "reproducibility_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing-official", action="store_true")
    args = parser.parse_args()
    report = audit(allow_missing_official=args.allow_missing_official)
    print(json.dumps(report["summary"], indent=2))
    raise SystemExit(1 if report["critical_error"] else 0)


if __name__ == "__main__":
    main()
