"""
Regression test: smoke runs must not overwrite full research outputs.

A ``--smoke`` invocation of any benchmark runner must redirect every
output artefact into an isolated ``smoke/`` sub-folder so a completed
full benchmark can never be clobbered by a quick smoke run.

This guards the fix for the overwrite risk where all three runners
previously wrote their smoke output to the same paths as the full run.
"""

from __future__ import annotations

from pathlib import Path

import run_latency_benchmark as latency_runner
import run_rq1_full_benchmark as rq1_runner
import run_thermal_tradeoff_sensitivity as c2_runner


def _assert_isolated(
    module,
    attribute_names: tuple[str, ...],
    label: str,
) -> None:
    before = {
        name: getattr(module, name) for name in attribute_names
    }
    full_directory = Path(module.OUTPUT_DIRECTORY)

    module.isolate_smoke_outputs()

    for name, original in before.items():
        redirected = Path(getattr(module, name))

        if redirected == original:
            raise AssertionError(
                f"{label}: {name} was not redirected for a smoke run."
            )

        if redirected.parent.name != "smoke":
            raise AssertionError(
                f"{label}: {name} is not inside a smoke folder "
                f"({redirected})."
            )

        if redirected.name != Path(original).name:
            raise AssertionError(
                f"{label}: {name} file name changed "
                f"({Path(original).name} -> {redirected.name})."
            )

        if Path(original).parent == redirected.parent:
            raise AssertionError(
                f"{label}: {name} still shares the full-run folder."
            )

    if Path(module.OUTPUT_DIRECTORY).parent != full_directory:
        raise AssertionError(
            f"{label}: the smoke directory is not nested under the "
            "full-run directory."
        )


def main() -> None:
    _assert_isolated(
        latency_runner,
        ("RAW_FILE", "SUMMARY_FILE", "RUN_CONFIG_FILE"),
        "latency",
    )

    _assert_isolated(
        rq1_runner,
        (
            "RAW_RUNS_FILE",
            "PLANNER_SCENARIO_FILE",
            "OVERALL_PLANNER_FILE",
            "FAILURE_CASES_FILE",
            "RUN_CONFIG_FILE",
        ),
        "rq1",
    )

    _assert_isolated(
        c2_runner,
        (
            "RAW_RUNS_FILE",
            "PARETO_FRONTS_FILE",
            "INSTANCE_SUMMARY_FILE",
            "RUN_CONFIG_FILE",
            "CHART_WES_FILE",
            "CHART_TRADEOFF_FILE",
        ),
        "c2",
    )

    print(
        "PASS: smoke runs write to isolated smoke folders and never "
        "overwrite full research outputs."
    )


if __name__ == "__main__":
    main()
