"""Pytest wrappers around the original script-style checks.

Eleven of the repository's ``test_*.py`` files are executable scripts:
they expose ``main()`` and ``raise`` on failure, so pytest never
collected them.  Rather than rewriting them (and risking a change in
what they actually assert) this module runs each ``main()`` unchanged
and converts any failure into an ordinary pytest failure.

Two rules make the wrappers safe:

* every check runs inside a throw-away working directory that contains
  copies of the read-only inputs it needs, so a script that writes
  ``results/...`` can never touch the frozen research artefacts;
* the previous working directory is restored afterwards.

The checks are kept as-is on purpose.  They are the historical evidence
that the frozen numbers still reproduce, so weakening them here would
defeat their purpose.
"""

from __future__ import annotations

import importlib
import shutil
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# Read-only inputs the legacy scripts load with relative paths.
DATA_FILES = (
    "poi_data_calculated_final.csv",
    "walking_time_matrix_final.csv",
)

RESULT_INPUTS = (
    "results/static_utility/initial_routes_by_seed.json",
    "results/weather_ablation/weather_ablation_raw_90_pairs.csv",
)


LEGACY_CHECKS = (
    pytest.param(
        "test_smoke_output_isolation",
        id="smoke-output-isolation",
    ),
    pytest.param(
        "test_application_services",
        id="application-services",
    ),
    pytest.param(
        "test_itinerary_service",
        id="itinerary-service",
    ),
    pytest.param(
        "test_scenario_states",
        id="scenario-states",
    ),
    pytest.param(
        "test_thermal_selection_contract",
        id="thermal-selection-contract",
    ),
    pytest.param(
        "test_adaptive_without_thermal",
        id="adaptive-without-thermal",
    ),
    pytest.param(
        "test_greedy_repair",
        id="greedy-repair",
    ),
    pytest.param(
        "test_tripmates_lite",
        id="tripmates-lite",
    ),
    pytest.param(
        "test_static_utility_reproducibility",
        marks=pytest.mark.slow,
        id="static-utility-reproducibility",
    ),
    pytest.param(
        "test_benchmark_determinism",
        marks=pytest.mark.slow,
        id="benchmark-determinism",
    ),
    pytest.param(
        "test_weather_ablation_seed",
        marks=pytest.mark.slow,
        id="weather-ablation-seed",
    ),
)


def _build_sandbox(destination: Path) -> Path:
    """Mirror the read-only inputs into an isolated working directory."""

    sandbox = destination / "repo"
    sandbox.mkdir(parents=True, exist_ok=True)

    for name in DATA_FILES:
        source = REPOSITORY_ROOT / name
        if not source.is_file():
            pytest.skip(f"Missing input data file: {name}")
        shutil.copy2(source, sandbox / name)

    for relative in RESULT_INPUTS:
        source = REPOSITORY_ROOT / relative
        if not source.is_file():
            pytest.skip(f"Missing frozen result file: {relative}")

        target = sandbox / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    # Pre-create the output roots so a script that writes into
    # ``results/`` does not have to guess whether they exist.
    for relative in (
        "results/scenario_states",
        "results/weather_ablation_v2",
        "results/static_utility",
    ):
        (sandbox / relative).mkdir(parents=True, exist_ok=True)

    return sandbox


def _snapshot_module_constants(module) -> dict[str, object]:
    return {
        name: getattr(module, name)
        for name in dir(module)
        if name.isupper() and not name.startswith("_")
    }


def _restore_module_constants(module, snapshot: dict[str, object]) -> None:
    for name, value in snapshot.items():
        setattr(module, name, value)


@pytest.mark.parametrize("module_name", LEGACY_CHECKS)
def test_legacy_script_check(module_name, monkeypatch, tmp_path):
    module = importlib.import_module(module_name)

    main = getattr(module, "main", None)
    assert callable(main), f"{module_name} does not expose a main() function."

    sandbox = _build_sandbox(tmp_path)

    # Some scripts mutate module-level output paths; restore them so the
    # rest of the session sees the official directories.
    snapshot = _snapshot_module_constants(module)

    monkeypatch.chdir(sandbox)

    try:
        main()
    except SystemExit as error:
        code = error.code
        if code not in (0, None):
            pytest.fail(
                f"{module_name}.main() exited with code {code!r}."
            )
    finally:
        _restore_module_constants(module, snapshot)
