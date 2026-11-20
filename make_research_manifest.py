"""
Generate the machine-readable research manifest for the new C2
experiment area.

The manifest records that the original 90-pair thermal ablation is the
confirmatory, fixed-policy result; that its observed outcome was no
selected-route change and no WES difference; that the C2 sensitivity
analysis is additional and exploratory; and that the C2 analysis must
not replace the original result.

It reads the frozen original summary (read-only) so the recorded facts
are sourced from the preserved artifact rather than hard-coded, and it
records fingerprints of the frozen files to demonstrate that they were
not modified.

Usage:
    python make_research_manifest.py
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmark_common import (
    MATRIX_FILE,
    POI_FILE,
    file_fingerprint,
    utc_timestamp,
    write_json,
)


ORIGINAL_RESULT_DIRECTORY = Path(
    "results/weather_ablation"
)
ORIGINAL_SUMMARY_FILE = (
    ORIGINAL_RESULT_DIRECTORY
    / "weather_ablation_statistical_summary.json"
)
ORIGINAL_RAW_FILE = (
    ORIGINAL_RESULT_DIRECTORY
    / "weather_ablation_raw_90_pairs.csv"
)
ORIGINAL_CONFIG_FILE = (
    ORIGINAL_RESULT_DIRECTORY
    / "weather_ablation_run_config.json"
)

NEW_EXPERIMENT_DIRECTORY = Path(
    "results/thermal_tradeoff_sensitivity"
)
MANIFEST_FILE = (
    NEW_EXPERIMENT_DIRECTORY / "experiment_manifest.json"
)

# ----------------------------------------------------------------
# Version 2 of the C2 sensitivity run.
#
# V1 mixed an alpha-sensitive generation-zero selection with an
# alpha-independent in-loop stagnation selection, which made the GA
# search trajectory unintentionally alpha-dependent.  V2 pins every
# internal search decision to one explicit frozen policy so only the
# final route selection depends on alpha.  V1 is preserved unchanged
# and superseded.
# ----------------------------------------------------------------

V2_EXPERIMENT_DIRECTORY = Path(
    "results/thermal_tradeoff_sensitivity_alpha_independent_v2"
)
V2_MANIFEST_FILE = (
    V2_EXPERIMENT_DIRECTORY / "experiment_manifest.json"
)

V2_SEARCH_POLICY_MODE = (
    "alpha_independent_frozen_policy"
)
V2_FINAL_SELECTION_POLICY_MODE = (
    "alpha_sensitive_exploratory"
)


def build_v2_manifest() -> dict:
    return {
        "manifest_version": 2,
        "generated_utc": utc_timestamp(),

        "experiment": {
            "name": (
                "C2 thermal trade-off sensitivity "
                "(alpha-independent search)"
            ),
            "directory": str(
                V2_EXPERIMENT_DIRECTORY
            ),
            "classification": (
                "exploratory sensitivity analysis"
            ),
            "is_confirmatory": False,
            "runner": (
                "run_thermal_tradeoff_sensitivity.py"
            ),
            "analysis": (
                "analyze_thermal_tradeoff_sensitivity.py"
            ),
        },

        # Required provenance fields.
        "search_policy_mode": V2_SEARCH_POLICY_MODE,
        "final_selection_policy_mode": (
            V2_FINAL_SELECTION_POLICY_MODE
        ),
        "supersedes": (
            "results/thermal_tradeoff_sensitivity/"
        ),
        "is_confirmatory": False,

        "design": {
            "alpha_scope": (
                "alpha affects ONLY the final route selection; it "
                "does not affect GA search evolution, stagnation "
                "detection, convergence generation or the evaluated "
                "Pareto-front population."
            ),
            "search_policy": (
                "one explicit frozen default policy, used for the "
                "generation-zero progress reference, the in-loop "
                "stagnation comparison and the convergence "
                "generation"
            ),
            "final_selection_policy": (
                "the requested C2 alpha with the C2 priority and "
                "guardrails"
            ),
            "regression_test": (
                "test_c2_alpha_independence.py"
            ),
        },

        "superseded_experiment": {
            "directory": (
                "results/thermal_tradeoff_sensitivity/"
            ),
            "reason": (
                "Its search was unintentionally alpha-dependent "
                "because the generation-zero selection used the "
                "requested alpha while the in-loop stagnation "
                "selection fell back to the frozen default."
            ),
            "preserved_unchanged": True,
        },

        "original_confirmatory_result": {
            "directory": "results/weather_ablation/",
            "statement": (
                "The frozen 90-pair thermal ablation remains the "
                "confirmatory fixed-policy result and is not "
                "replaced by this sensitivity analysis."
            ),
            "frozen_artifacts": [
                file_fingerprint(ORIGINAL_SUMMARY_FILE),
                file_fingerprint(ORIGINAL_RAW_FILE),
                file_fingerprint(ORIGINAL_CONFIG_FILE),
            ],
        },

        "frozen_inputs": {
            "poi_dataset": file_fingerprint(POI_FILE),
            "walking_matrix": file_fingerprint(MATRIX_FILE),
        },
    }


def main() -> None:
    original_summary = json.loads(
        ORIGINAL_SUMMARY_FILE.read_text(
            encoding="utf-8"
        )
    )

    original_config = json.loads(
        ORIGINAL_CONFIG_FILE.read_text(
            encoding="utf-8"
        )
    )

    manifest = {
        "manifest_version": 1,
        "generated_utc": utc_timestamp(),

        "new_experiment": {
            "name": "C2 thermal trade-off sensitivity",
            "directory": str(NEW_EXPERIMENT_DIRECTORY),
            "classification": (
                "exploratory sensitivity analysis"
            ),
            "is_confirmatory": False,
            "purpose": (
                "Quantify the utility-exposure trade-off under "
                "predefined utility-retention policies."
            ),
            "runner": (
                "run_thermal_tradeoff_sensitivity.py"
            ),
            "analysis": (
                "analyze_thermal_tradeoff_sensitivity.py"
            ),
        },

        "original_confirmatory_experiment": {
            "name": original_config.get(
                "experiment",
                "Thermal-awareness ablation",
            ),
            "directory": str(
                ORIGINAL_RESULT_DIRECTORY
            ),
            "classification": (
                "confirmatory fixed-policy result"
            ),
            "planners": original_config.get(
                "planners",
                [],
            ),
            "paired_result_count": original_summary.get(
                "pair_count"
            ),
            "instance_count": original_summary.get(
                "instance_count"
            ),
            "seed_count": original_summary.get(
                "seed_count"
            ),
            "observed_outcome": {
                "route_change": (
                    "no selected-route change in any pair"
                ),
                "wes_difference": (
                    "no weather-weighted exposure "
                    "difference in any pair"
                ),
                "overall_wes_p_value": (
                    original_summary.get(
                        "overall_wes_p_value"
                    )
                ),
                "overall_wes_effect_size": (
                    original_summary.get(
                        "overall_wes_effect"
                    )
                ),
                "test_status": original_summary.get(
                    "overall_wes_status"
                ),
                "conclusion": original_summary.get(
                    "conclusion"
                ),
            },
            "frozen_artifacts": [
                file_fingerprint(ORIGINAL_SUMMARY_FILE),
                file_fingerprint(ORIGINAL_RAW_FILE),
                file_fingerprint(ORIGINAL_CONFIG_FILE),
            ],
        },

        "relationship": {
            "c2_is_additional": True,
            "c2_must_not_replace_original": True,
            "statement": (
                "The original 90-pair thermal ablation is a "
                "confirmatory fixed-policy result. Its observed "
                "outcome was no selected-route change and no WES "
                "difference. The C2 sensitivity analysis is "
                "additional and exploratory and must not replace "
                "the original result."
            ),
            "original_files_modified": False,
            "note": (
                "No original raw CSV or summary file is edited, "
                "overwritten or relabelled by this work."
            ),
        },

        "frozen_inputs": {
            "poi_dataset": file_fingerprint(POI_FILE),
            "walking_matrix": file_fingerprint(MATRIX_FILE),
        },
    }

    write_json(MANIFEST_FILE, manifest)

    V2_EXPERIMENT_DIRECTORY.mkdir(
        parents=True, exist_ok=True
    )
    write_json(V2_MANIFEST_FILE, build_v2_manifest())

    print(
        "Wrote research manifest:",
        MANIFEST_FILE.resolve(),
    )
    print(
        "Wrote V2 research manifest:",
        V2_MANIFEST_FILE.resolve(),
    )


if __name__ == "__main__":
    main()
