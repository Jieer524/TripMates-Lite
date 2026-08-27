import os
import sys
import io
import json
import math
import hashlib
import subprocess
import platform
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

REPORT_VERSION = "2.0.0"

EXPECTED_POI_COUNT = 50
EXPECTED_IDS = list(range(1, EXPECTED_POI_COUNT + 1))

UTILITY_TOLERANCE = 1e-6

# Exact formula verification using original high-precision
# coordinates.
MATRIX_EXACT_TOLERANCE = 1e-6

# Secondary verification using the final CSV coordinates.
# The final CSV stores coordinates rounded to 6 decimals.
# The tolerance is therefore intentionally larger.
MATRIX_ROUNDED_COORD_TOLERANCE = 0.002

EARTH_RADIUS_METRES = 6371000.0
WALKING_SPEED_MPS = 1.4

PROJECT_BUDGET_RM = 160.0

REPOSITORY_URL = "https://github.com/Jieer524/TripMates-Lite"

# Commit from which the original high-precision POI dataset
# is recovered.
SOURCE_DATASET_COMMIT = "b38ee40500978f3f5a02ec6f0814de8b6ff326a1"

SOURCE_DATASET_PATH = "latest/poi_data_final_v2.csv"


# ============================================================
# HELPERS
# ============================================================

def get_git_commit():
    """Return the current repository commit SHA."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True
        ).strip()
    except Exception:
        return "NOT_VERIFIABLE"


def get_git_branch():
    """Return the current Git branch."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True
        ).strip()
    except Exception:
        return "NOT_VERIFIABLE"


def sha256_file(path):
    """Calculate SHA-256 hash of a file."""
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return h.hexdigest()


def calculate_walking_matrix(coords):
    """
    Recalculate walking-time matrix using:

        d = R * sqrt(dx^2 + dy^2)

    where:

        dx = Δlongitude * cos(mean latitude)
        dy = Δlatitude

    and:

        walking_time_minutes =
            distance_metres / walking_speed / 60

    Results are rounded to 6 decimal places to match
    the stored walking-time matrix.
    """

    n = len(coords)

    result = np.zeros((n, n), dtype=float)

    for i in range(n):

        lat_i_rad = math.radians(float(coords[i][0]))
        lon_i_rad = math.radians(float(coords[i][1]))

        for j in range(i + 1, n):

            lat_j_rad = math.radians(float(coords[j][0]))
            lon_j_rad = math.radians(float(coords[j][1]))

            mean_lat = (
                lat_i_rad + lat_j_rad
            ) / 2.0

            dx = (
                lon_j_rad - lon_i_rad
            ) * math.cos(mean_lat)

            dy = (
                lat_j_rad - lat_i_rad
            )

            distance_metres = (
                EARTH_RADIUS_METRES
                * math.sqrt(dx ** 2 + dy ** 2)
            )

            walking_minutes = (
                distance_metres
                / WALKING_SPEED_MPS
                / 60.0
            )

            walking_minutes = round(
                walking_minutes,
                6
            )

            result[i, j] = walking_minutes
            result[j, i] = walking_minutes

    return result


def validate_matrix_formula(
    stored_matrix,
    coords,
    tolerance
):
    """Compare stored matrix with independently recalculated matrix."""

    recalculated_matrix = calculate_walking_matrix(coords)

    difference = np.abs(
        recalculated_matrix - stored_matrix
    )

    max_difference = float(
        np.max(difference)
    )

    mean_difference = float(
        np.mean(difference)
    )

    mismatch_count = int(
        np.sum(difference > tolerance) // 2
    )

    return {
        "maximum_absolute_difference_minutes":
            max_difference,

        "mean_absolute_difference_minutes":
            mean_difference,

        "maximum_absolute_difference_seconds":
            max_difference * 60.0,

        "mismatch_count":
            mismatch_count,

        "tolerance_minutes":
            tolerance,

        "status":
            "PASS" if mismatch_count == 0
            else "FAIL"
    }


# ============================================================
# MAIN AUDIT
# ============================================================

def run_audit():

    print("=" * 70)
    print("TripMates Lite Dataset Validation")
    print("=" * 70)

    results = {}

    # --------------------------------------------------------
    # 1. FILE PATHS
    # --------------------------------------------------------

    input_path = Path("poi_data_final_v2.csv")
    calc_path = Path("poi_data_calculated_final.csv")
    matrix_path = Path("walking_time_matrix_final.csv")
    report_path = Path("dataset_validation_report.json")

    # --------------------------------------------------------
    # 2. LOAD FILES
    # --------------------------------------------------------

    if not calc_path.exists():
        raise FileNotFoundError(
            f"Missing file: {calc_path}"
        )

    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Missing file: {matrix_path}"
        )

    df_calc = pd.read_csv(calc_path)
    df_matrix = pd.read_csv(matrix_path)

    # Original high-precision dataset from Git history.
    df_orig = None
    raw_orig_bytes = None

    try:

        out_orig = subprocess.check_output(
            [
                "git",
                "show",
                f"{SOURCE_DATASET_COMMIT}:{SOURCE_DATASET_PATH}"
            ],
            text=True
        )

        df_orig = pd.read_csv(
            io.StringIO(out_orig)
        )

        raw_orig_bytes = subprocess.check_output(
            [
                "git",
                "show",
                f"{SOURCE_DATASET_COMMIT}:{SOURCE_DATASET_PATH}"
            ]
        )

    except Exception as e:

        print(
            "WARNING: Could not recover original dataset "
            f"from Git: {e}"
        )

    print("Files loaded successfully.")

    # --------------------------------------------------------
    # 3. METADATA
    # --------------------------------------------------------

    validation_metadata = {

        "report_version":
            REPORT_VERSION,

        "validation_timestamp_utc":
            datetime.now(timezone.utc).isoformat(),

        "repository": {
            "url":
                REPOSITORY_URL,

            "branch":
                get_git_branch(),

            "validation_commit_sha":
                get_git_commit(),

            "source_dataset_commit_sha":
                SOURCE_DATASET_COMMIT,

            "source_dataset_git_path":
                SOURCE_DATASET_PATH
        },

        "environment": {
            "python_version":
                sys.version.split()[0],

            "pandas_version":
                pd.__version__,

            "numpy_version":
                np.__version__,

            "operating_system":
                platform.platform()
        }
    }

    # --------------------------------------------------------
    # 4. DATASET / POI CATALOGUE VALIDATION
    # --------------------------------------------------------

    row_count = len(df_calc)

    poi_ids = df_calc["poi_id"].tolist()

    missing_ids = sorted(
        list(
            set(EXPECTED_IDS) -
            set(poi_ids)
        )
    )

    duplicate_ids = (
        df_calc[
            df_calc["poi_id"].duplicated()
        ]["poi_id"]
        .tolist()
    )

    unexpected_ids = sorted(
        list(
            set(poi_ids) -
            set(EXPECTED_IDS)
        )
    )

    required_fields_planner = [
        "poi_id",
        "name",
        "category",
        "latitude",
        "longitude",
        "zone",
        "mandatory",
        "off_day",
        "is_24_hours",
        "opening_time_24h",
        "closing_time_24h",
        "closes_next_day",
        "visit_duration_min",
        "admission_cost_rm",
        "utility_score",
        "indoor_ratio",
        "shelter_ratio"
    ]

    all_fields_present = list(
        df_calc.columns
    )

    missing_planner_cols = [
        c
        for c in required_fields_planner
        if c not in all_fields_present
    ]

    missing_values_per_col = (
        df_calc.isna()
        .sum()
        .to_dict()
    )

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    lat_numeric = pd.to_numeric(
        df_calc["latitude"],
        errors="coerce"
    )

    lon_numeric = pd.to_numeric(
        df_calc["longitude"],
        errors="coerce"
    )

    lat_invalid = (
        lat_numeric.isna()
        | (lat_numeric < -90)
        | (lat_numeric > 90)
    )

    lon_invalid = (
        lon_numeric.isna()
        | (lon_numeric < -180)
        | (lon_numeric > 180)
    )

    # KL geographic plausibility check.
    kl_lat_invalid = ~(
        (lat_numeric >= 3.0)
        & (lat_numeric <= 3.3)
    )

    kl_lon_invalid = ~(
        (lon_numeric >= 101.5)
        & (lon_numeric <= 101.8)
    )

    # --------------------------------------------------------
    # Rating
    # --------------------------------------------------------

    rating_numeric = pd.to_numeric(
        df_calc["google_rating"],
        errors="coerce"
    )

    rating_invalid_range = (
        rating_numeric.isna()
        | (rating_numeric < 0)
        | (rating_numeric > 5)
    )

    recalculated_rating_norm = (
        2.0 * rating_numeric
    )

    stored_rating_norm = pd.to_numeric(
        df_calc["rating_norm"],
        errors="coerce"
    )

    rating_norm_diff = np.abs(
        recalculated_rating_norm -
        stored_rating_norm
    )

    rating_norm_mismatches = int(
        (rating_norm_diff > UTILITY_TOLERANCE)
        .sum()
    )

    # --------------------------------------------------------
    # Review count
    # --------------------------------------------------------

    review_numeric = pd.to_numeric(
        df_calc["review_count"],
        errors="coerce"
    )

    review_invalid = (
        review_numeric.isna()
        | (review_numeric < 0)
    )

    max_reviews = float(
        review_numeric.max()
    )

    recalculated_review_norm = (
        10.0
        * np.log10(review_numeric + 1)
        / np.log10(max_reviews + 1)
    )

    stored_review_norm = pd.to_numeric(
        df_calc["review_norm"],
        errors="coerce"
    )

    review_norm_diff = np.abs(
        recalculated_review_norm -
        stored_review_norm
    )

    review_norm_mismatches = int(
        (review_norm_diff > UTILITY_TOLERANCE)
        .sum()
    )

    # --------------------------------------------------------
    # Zone
    # --------------------------------------------------------

    zone_map = {
        "A": 10.0,
        "B": 8.0,
        "C": 6.0
    }

    zones = (
        df_calc["zone"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    invalid_zones = sorted(
        list(
            set(
                z
                for z in zones
                if z not in zone_map
            )
        )
    )

    recalculated_zone_score = zones.map(
        zone_map
    )

    stored_zone_score = pd.to_numeric(
        df_calc["zone_score"],
        errors="coerce"
    )

    zone_score_diff = np.abs(
        recalculated_zone_score -
        stored_zone_score
    )

    zone_score_mismatches = int(
        (zone_score_diff > UTILITY_TOLERANCE)
        .sum()
    )

    # --------------------------------------------------------
    # Utility
    # --------------------------------------------------------

    recalculated_utility = (
        recalculated_rating_norm
        + recalculated_review_norm
        + recalculated_zone_score
    ) / 3.0

    stored_utility = pd.to_numeric(
        df_calc["utility_score"],
        errors="coerce"
    )

    utility_diff = np.abs(
        recalculated_utility -
        stored_utility
    )

    utility_mismatches = int(
        (utility_diff > UTILITY_TOLERANCE)
        .sum()
    )

    min_utility = float(
        stored_utility.min()
    )

    max_utility = float(
        stored_utility.max()
    )

    # --------------------------------------------------------
    # Visit duration
    # --------------------------------------------------------

    duration_numeric = pd.to_numeric(
        df_calc["visit_duration_min"],
        errors="coerce"
    )

    duration_invalid = (
        duration_numeric.isna()
        | (duration_numeric <= 0)
    )

    # --------------------------------------------------------
    # Admission cost
    # --------------------------------------------------------

    cost_numeric = pd.to_numeric(
        df_calc["admission_cost_rm"],
        errors="coerce"
    )

    cost_invalid = (
        cost_numeric.isna()
        | (cost_numeric < 0)
    )

    cost_exceeding_budget = (
        df_calc[
            cost_numeric > PROJECT_BUDGET_RM
        ][
            [
                "poi_id",
                "name",
                "admission_cost_rm"
            ]
        ]
        .to_dict(orient="records")
    )

    # --------------------------------------------------------
    # Opening hours
    # --------------------------------------------------------

    from planning_core import (
        parse_clock,
        parse_yes_no,
        parse_off_days
    )

    opening_hour_errors = []

    for _, row in df_calc.iterrows():

        pid = row["poi_id"]

        try:

            op_m = parse_clock(
                row["opening_time_24h"]
            )

            cl_m = parse_clock(
                row["closing_time_24h"]
            )

            is_24 = parse_yes_no(
                row["is_24_hours"],
                "is_24_hours"
            )

            c_next = parse_yes_no(
                row["closes_next_day"],
                "closes_next_day"
            )

            parse_off_days(
                row["off_day"]
            )

            if (
                not is_24
                and not c_next
                and op_m >= cl_m
            ):

                opening_hour_errors.append(
                    f"POI {pid} "
                    f"({row['name']}): "
                    f"opening {row['opening_time_24h']} "
                    f">= closing "
                    f"{row['closing_time_24h']}"
                )

        except Exception as err:

            opening_hour_errors.append(
                f"POI {pid} "
                f"({row['name']}): {err}"
            )

    # --------------------------------------------------------
    # Indoor / shelter
    # --------------------------------------------------------

    indoor_numeric = pd.to_numeric(
        df_calc["indoor_ratio"],
        errors="coerce"
    )

    shelter_numeric = pd.to_numeric(
        df_calc["shelter_ratio"],
        errors="coerce"
    )

    indoor_invalid = (
        indoor_numeric.isna()
        | (indoor_numeric < 0)
        | (indoor_numeric > 1)
    )

    shelter_invalid = (
        shelter_numeric.isna()
        | (shelter_numeric < 0)
        | (shelter_numeric > 1)
    )

    # --------------------------------------------------------
    # DATASET VALIDATION RESULT
    # --------------------------------------------------------

    dataset_passed = (
        row_count == EXPECTED_POI_COUNT
        and len(missing_ids) == 0
        and len(duplicate_ids) == 0
        and len(unexpected_ids) == 0
        and len(missing_planner_cols) == 0
        and sum(missing_values_per_col.values()) == 0
        and int(lat_invalid.sum()) == 0
        and int(lon_invalid.sum()) == 0
        and int(rating_invalid_range.sum()) == 0
        and int(review_invalid.sum()) == 0
        and len(invalid_zones) == 0
        and rating_norm_mismatches == 0
        and review_norm_mismatches == 0
        and zone_score_mismatches == 0
        and utility_mismatches == 0
        and int(duration_invalid.sum()) == 0
        and int(cost_invalid.sum()) == 0
        and len(cost_exceeding_budget) == 0
        and len(opening_hour_errors) == 0
        and int(indoor_invalid.sum()) == 0
        and int(shelter_invalid.sum()) == 0
    )

    # --------------------------------------------------------
    # 5. DERIVED POI FILE VALIDATION
    # --------------------------------------------------------

    derived_validation = {
        "source_available": df_orig is not None
    }

    if df_orig is not None:

        derived_validation[
            "row_count_match"
        ] = (
            len(df_orig) ==
            len(df_calc)
        )

        derived_validation[
            "poi_id_exact_match"
        ] = (
            df_orig["poi_id"].tolist()
            ==
            df_calc["poi_id"].tolist()
        )

        derived_validation[
            "coordinate_rounding"
        ] = {}

        for col in [
            "latitude",
            "longitude"
        ]:

            max_diff = float(
                np.abs(
                    df_orig[col] -
                    df_calc[col]
                ).max()
            )

            derived_validation[
                "coordinate_rounding"
            ][
                f"{col}_max_difference"
            ] = max_diff

        for col in [
            "rating_norm",
            "review_norm",
            "zone_score",
            "utility_score"
        ]:

            max_diff = float(
                np.abs(
                    df_orig[col] -
                    df_calc[col]
                ).max()
            )

            derived_validation[
                f"{col}_max_difference"
            ] = max_diff

    # --------------------------------------------------------
    # 6. WALKING-TIME MATRIX VALIDATION
    # --------------------------------------------------------

    matrix_rows, matrix_cols = (
        df_matrix.shape
    )

    # IMPORTANT:
    #
    # The CSV has:
    #
    #   1 poi_id column
    #   50 actual matrix columns
    #
    # Therefore:
    #
    #   CSV shape = 50 x 51
    #   actual matrix shape = 50 x 50

    mat_poi_col = (
        df_matrix["poi_id"]
        .tolist()
    )

    mat_val_cols = list(
        df_matrix.columns[1:]
    )

    mat_values = (
        df_matrix
        .drop(columns=["poi_id"])
        .to_numpy(dtype=float)
    )

    actual_matrix_shape = [
        int(mat_values.shape[0]),
        int(mat_values.shape[1])
    ]

    expected_matrix_shape = [
        EXPECTED_POI_COUNT,
        EXPECTED_POI_COUNT
    ]

    csv_storage_shape = [
        int(matrix_rows),
        int(matrix_cols)
    ]

    row_ids_valid = (
        mat_poi_col ==
        EXPECTED_IDS
    )

    col_ids_valid = (
        mat_val_cols ==
        [
            str(i)
            for i in EXPECTED_IDS
        ]
    )

    # --------------------------------------------------------
    # Numeric validity
    # --------------------------------------------------------

    finite_vals = bool(
        np.isfinite(mat_values).all()
    )

    nan_count = int(
        np.isnan(mat_values).sum()
    )

    inf_count = int(
        np.isinf(mat_values).sum()
    )

    neg_count = int(
        (mat_values < 0).sum()
    )

    # --------------------------------------------------------
    # Diagonal
    # --------------------------------------------------------

    diagonal = np.diag(
        mat_values
    )

    zero_diagonal = bool(
        np.allclose(
            diagonal,
            0.0,
            atol=1e-12
        )
    )

    max_diag_abs = float(
        np.max(
            np.abs(diagonal)
        )
    )

    # --------------------------------------------------------
    # Non-diagonal
    # --------------------------------------------------------

    non_diag_mask = ~np.eye(
        EXPECTED_POI_COUNT,
        dtype=bool
    )

    non_diag_vals = (
        mat_values[non_diag_mask]
    )

    positive_non_diag = bool(
        np.all(non_diag_vals > 0)
    )

    min_non_diag = float(
        non_diag_vals.min()
    )

    max_non_diag = float(
        non_diag_vals.max()
    )

    zero_non_diag_count = int(
        (non_diag_vals == 0).sum()
    )

    # --------------------------------------------------------
    # Symmetry
    # --------------------------------------------------------

    asym_matrix = np.abs(
        mat_values -
        mat_values.T
    )

    max_asymmetry = float(
        np.max(asym_matrix)
    )

    asymmetric_pairs_count = int(
        (
            asym_matrix > 1e-12
        ).sum() // 2
    )

    is_symmetric = (
        max_asymmetry <= 1e-12
    )

    # --------------------------------------------------------
    # 7. MATRIX FORMULA VERIFICATION
    # --------------------------------------------------------

    matrix_formula_validation = {}

    # ========================================================
    # PRIMARY:
    # Original high-precision coordinates
    #
    # This is the authoritative formula verification.
    # ========================================================

    if df_orig is not None:

        original_coords = (
            df_orig[
                [
                    "latitude",
                    "longitude"
                ]
            ]
            .to_numpy(dtype=float)
        )

        primary_formula = (
            validate_matrix_formula(
                mat_values,
                original_coords,
                MATRIX_EXACT_TOLERANCE
            )
        )

        matrix_formula_validation[
            "primary_original_coordinates"
        ] = {
            "coordinate_source":
                "Original high-precision POI coordinates",

            **primary_formula,

            "interpretation":
                "Primary verification of the walking-time formula. "
                "The stored matrix should reproduce the documented "
                "equirectangular calculation from the original "
                "high-precision coordinates."
        }

    else:

        matrix_formula_validation[
            "primary_original_coordinates"
        ] = {
            "coordinate_source":
                "Original high-precision POI coordinates",

            "status":
                "NOT_VERIFIABLE",

            "reason":
                "Original source dataset could not be recovered "
                "from the specified Git commit."
        }

    # ========================================================
    # SECONDARY:
    # Final calculated CSV coordinates
    #
    # These coordinates are rounded to 6 decimal places.
    # ========================================================

    stored_coords = (
        df_calc[
            [
                "latitude",
                "longitude"
            ]
        ]
        .to_numpy(dtype=float)
    )

    secondary_formula = (
        validate_matrix_formula(
            mat_values,
            stored_coords,
            MATRIX_ROUNDED_COORD_TOLERANCE
        )
    )

    matrix_formula_validation[
        "secondary_final_csv_coordinates"
    ] = {
        "coordinate_source":
            "Coordinates stored in final calculated POI CSV",

        **secondary_formula,

        "coordinate_storage_precision":
            "6 decimal places",

        "interpretation":
            "This secondary check measures the numerical effect "
            "of coordinate rounding in the final CSV. Small "
            "differences are expected because the stored CSV "
            "coordinates have lower precision than the original "
            "source coordinates."
    }

    # --------------------------------------------------------
    # 8. MATRIX VALIDATION PASS/FAIL
    # --------------------------------------------------------

    matrix_shape_valid = (
        actual_matrix_shape ==
        expected_matrix_shape
    )

    matrix_validation_passed = (
        matrix_shape_valid
        and row_ids_valid
        and col_ids_valid
        and finite_vals
        and nan_count == 0
        and inf_count == 0
        and neg_count == 0
        and zero_diagonal
        and positive_non_diag
        and is_symmetric
        and (
            matrix_formula_validation[
                "primary_original_coordinates"
            ]["status"]
            == "PASS"
        )
    )

    # --------------------------------------------------------
    # 9. CROSS-FILE VALIDATION
    # --------------------------------------------------------

    cross_file_validation = {

        "calculated_poi_ids_match_expected":
            poi_ids == EXPECTED_IDS,

        "matrix_row_ids_match_poi_catalogue":
            row_ids_valid,

        "matrix_column_ids_match_poi_catalogue":
            col_ids_valid,

        "matrix_dimensions_match_catalogue":
            actual_matrix_shape ==
            [
                len(df_calc),
                len(df_calc)
            ]
    }

    # --------------------------------------------------------
    # 10. FILE INTEGRITY
    # --------------------------------------------------------

    sha256_hashes = {}

    for path in [
        input_path,
        calc_path,
        matrix_path
    ]:

        if path.exists():

            sha256_hashes[
                path.name
            ] = sha256_file(path)

    if raw_orig_bytes is not None:

        sha256_hashes[
            "original_poi_file_git_source"
        ] = hashlib.sha256(
            raw_orig_bytes
        ).hexdigest()

    # --------------------------------------------------------
    # 11. OVERALL VALIDATION
    # --------------------------------------------------------

    failed_checks = []
    warnings = []

    if not dataset_passed:
        failed_checks.append(
            "dataset_validation"
        )

    if not matrix_validation_passed:
        failed_checks.append(
            "matrix_validation"
        )

    if not cross_file_validation[
        "matrix_dimensions_match_catalogue"
    ]:
        failed_checks.append(
            "cross_file_matrix_dimensions"
        )

    if (
        matrix_formula_validation[
            "primary_original_coordinates"
        ]["status"]
        == "NOT_VERIFIABLE"
    ):
        warnings.append(
            "Primary matrix formula verification "
            "could not be performed."
        )

    secondary_status = (
        matrix_formula_validation[
            "secondary_final_csv_coordinates"
        ]["status"]
    )

    if secondary_status == "PASS":

        warnings.append(
            "The secondary matrix formula check uses "
            "6-decimal-place coordinates from the final CSV. "
            "Its small numerical difference is evaluated using "
            "the documented rounding tolerance."
        )

    overall_passed = (
        len(failed_checks) == 0
    )

    # --------------------------------------------------------
    # 12. BUILD FINAL JSON REPORT
    # --------------------------------------------------------

    report = {

        "validation_metadata":
            validation_metadata,

        "input_files": {

            "input_file":
                input_path.name,

            "calculated_poi_file":
                calc_path.name,

            "walking_matrix_file":
                matrix_path.name
        },

        "formula_configuration": {

            "utility_formula": {

                "rating_norm":
                    "2 * google_rating",

                "review_norm":
                    "10 * log10(review_count + 1) / "
                    "log10(max_review_count + 1)",

                "zone_score": {
                    "A": 10.0,
                    "B": 8.0,
                    "C": 6.0
                },

                "utility_score":
                    "(rating_norm + review_norm + zone_score) / 3",

                "formula_version":
                    "U1_equal_rating_review_zone"
            },

            "walking_formula": {

                "method":
                    "Equirectangular straight-line "
                    "distance approximation",

                "earth_radius_metres":
                    EARTH_RADIUS_METRES,

                "walking_speed_metres_per_second":
                    WALKING_SPEED_MPS,

                "walking_time":
                    "distance_metres / 1.4 / 60",

                "matrix_stored_decimal_places":
                    6
            },

            "numerical_tolerances": {

                "utility_tolerance":
                    UTILITY_TOLERANCE,

                "matrix_exact_tolerance_minutes":
                    MATRIX_EXACT_TOLERANCE,

                "matrix_rounded_coordinate_tolerance_minutes":
                    MATRIX_ROUNDED_COORD_TOLERANCE
            }
        },

        "schema_validation": {

            "row_count":
                row_count,

            "expected_row_count":
                EXPECTED_POI_COUNT,

            "missing_ids":
                missing_ids,

            "duplicate_ids":
                duplicate_ids,

            "unexpected_ids":
                unexpected_ids,

            "missing_planner_columns":
                missing_planner_cols,

            "missing_values":
                missing_values_per_col
        },

        "dataset_validation": {

            "coordinates": {

                "invalid_latitude_count":
                    int(lat_invalid.sum()),

                "invalid_longitude_count":
                    int(lon_invalid.sum()),

                "kl_latitude_outside_expected_range":
                    int(kl_lat_invalid.sum()),

                "kl_longitude_outside_expected_range":
                    int(kl_lon_invalid.sum())
            },

            "rating": {

                "invalid_range_count":
                    int(rating_invalid_range.sum()),

                "rating_norm_mismatch_count":
                    rating_norm_mismatches,

                "rating_norm_max_difference":
                    float(rating_norm_diff.max())
            },

            "review_count": {

                "invalid_count":
                    int(review_invalid.sum()),

                "maximum_review_count":
                    max_reviews,

                "review_norm_mismatch_count":
                    review_norm_mismatches,

                "review_norm_max_difference":
                    float(review_norm_diff.max())
            },

            "zone": {

                "invalid_zones":
                    invalid_zones,

                "zone_score_mismatch_count":
                    zone_score_mismatches,

                "zone_score_max_difference":
                    float(zone_score_diff.max())
            },

            "utility": {

                "utility_score_mismatch_count":
                    utility_mismatches,

                "utility_score_max_difference":
                    float(utility_diff.max()),

                "minimum_utility":
                    min_utility,

                "maximum_utility":
                    max_utility
            },

            "visit_duration": {

                "invalid_count":
                    int(duration_invalid.sum()),

                "minimum_minutes":
                    float(duration_numeric.min()),

                "maximum_minutes":
                    float(duration_numeric.max())
            },

            "admission_cost": {

                "invalid_count":
                    int(cost_invalid.sum()),

                "minimum_rm":
                    float(cost_numeric.min()),

                "maximum_rm":
                    float(cost_numeric.max()),

                "project_budget_rm":
                    PROJECT_BUDGET_RM,

                "exceeding_budget":
                    cost_exceeding_budget
            },

            "opening_hours": {

                "error_count":
                    len(opening_hour_errors),

                "errors":
                    opening_hour_errors
            },

            "exposure_ratios": {

                "indoor_invalid_count":
                    int(indoor_invalid.sum()),

                "shelter_invalid_count":
                    int(shelter_invalid.sum())
            },

            "passed":
                dataset_passed
        },

        "derived_asset_validation":
            derived_validation,

        "matrix_validation": {

            "csv_storage_shape":
                csv_storage_shape,

            "csv_storage_format":
                "poi_id column + 50 numeric POI columns",

            "actual_matrix_shape":
                actual_matrix_shape,

            "expected_matrix_shape":
                expected_matrix_shape,

            "shape_valid":
                matrix_shape_valid,

            "row_ids_exactly_1_to_50":
                row_ids_valid,

            "column_ids_exactly_1_to_50":
                col_ids_valid,

            "finite_values":
                finite_vals,

            "nan_count":
                nan_count,

            "infinity_count":
                inf_count,

            "negative_value_count":
                neg_count,

            "zero_diagonal":
                zero_diagonal,

            "maximum_diagonal_absolute_value":
                max_diag_abs,

            "positive_non_diagonal":
                positive_non_diag,

            "zero_non_diagonal_count":
                zero_non_diag_count,

            "minimum_non_diagonal_minutes":
                min_non_diag,

            "maximum_non_diagonal_minutes":
                max_non_diag,

            "symmetric":
                is_symmetric,

            "maximum_asymmetry_minutes":
                max_asymmetry,

            "asymmetric_pairs_count":
                asymmetric_pairs_count,

            "passed":
                matrix_validation_passed
        },

        "matrix_formula_validation":
            matrix_formula_validation,

        "cross_file_validation":
            cross_file_validation,

        "file_integrity": {

            "sha256":
                sha256_hashes
        },

        "overall_validation": {

            "passed":
                overall_passed,

            "failed_checks":
                failed_checks,

            "warnings":
                warnings
        }
    }

    # --------------------------------------------------------
    # 13. WRITE JSON REPORT
    # --------------------------------------------------------

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # 14. CONSOLE SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(
        f"Dataset validation: "
        f"{'PASS' if dataset_passed else 'FAIL'}"
    )

    print(
        f"Matrix shape: "
        f"{actual_matrix_shape} "
        f"(CSV storage: {csv_storage_shape})"
    )

    print(
        f"Matrix structural validation: "
        f"{'PASS' if matrix_validation_passed else 'FAIL'}"
    )

    primary = matrix_formula_validation[
        "primary_original_coordinates"
    ]

    secondary = matrix_formula_validation[
        "secondary_final_csv_coordinates"
    ]

    print(
        f"Primary matrix formula validation: "
        f"{primary['status']}"
    )

    print(
        f"  Maximum difference: "
        f"{primary.get('maximum_absolute_difference_minutes', 'N/A')} "
        f"minutes"
    )

    print(
        f"Secondary rounded-coordinate validation: "
        f"{secondary['status']}"
    )

    print(
        f"  Maximum difference: "
        f"{secondary['maximum_absolute_difference_minutes']} "
        f"minutes"
    )

    print(
        f"Overall validation: "
        f"{'PASS' if overall_passed else 'FAIL'}"
    )

    print()
    print(
        f"Report written to: "
        f"{report_path.resolve()}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_audit()