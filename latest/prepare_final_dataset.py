"""
Prepare the final TripMates Lite experimental dataset.

This script calculates:

1. rating_norm
2. review_norm
3. zone_score
4. utility_score
5. utility_formula_version
6. 50 x 50 walking-time matrix
7. validation results
8. SHA-256 file hashes

The input CSV should contain the manually collected and verified POI
attributes. The script creates new output files and does not overwrite
the original input file.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ================================================================
# Configuration
# ================================================================

INPUT_POI_FILE = Path("poi_data_final_v2.csv")

OUTPUT_POI_FILE = Path("poi_data_calculated_final.csv")
OUTPUT_MATRIX_FILE = Path("walking_time_matrix_final.csv")
OUTPUT_VALIDATION_FILE = Path("dataset_validation_report.json")

EXPECTED_POI_COUNT = 50

# Walking assumptions from Chapter 3
EARTH_RADIUS_METRES = 6_371_000.0
WALKING_SPEED_METRES_PER_SECOND = 1.4

# Utility configuration
ZONE_SCORES = {
    "A": 10.0,
    "B": 8.0,
    "C": 6.0,
}

UTILITY_FORMULA_VERSION = "U1_equal_rating_review_zone"

# Number of decimal places stored in output files
UTILITY_DECIMAL_PLACES = 6
MATRIX_DECIMAL_PLACES = 6


# ================================================================
# General helpers
# ================================================================

def calculate_sha256(file_path: Path) -> str:
    """Calculate a SHA-256 checksum for a file."""

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(65_536), b""):
            digest.update(block)

    return digest.hexdigest()


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
) -> None:
    """Raise an error when required columns are missing."""

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "The input dataset is missing these required columns: "
            f"{sorted(missing_columns)}"
        )


# ================================================================
# Load and validate manually collected data
# ================================================================

def load_poi_dataset(file_path: Path) -> pd.DataFrame:
    """Load and validate the manually collected POI attributes."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"POI dataset not found: {file_path.resolve()}"
        )

    dataframe = pd.read_csv(file_path)

    required_columns = {
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
        "google_rating",
        "review_count",
        "metadata_source",
        "metadata_collected_date",
        "source_url",
        "visit_duration_min",
        "visit_duration_class",
        "visit_duration_basis",
        "admission_cost_rm",
        "cost_source_url",
        "cost_verified_date",
        "indoor_ratio",
        "shelter_ratio",
        "exposure_ratio_basis",
    }

    require_columns(dataframe, required_columns)

    # ------------------------------------------------------------
    # POI ID validation
    # ------------------------------------------------------------

    dataframe["poi_id"] = pd.to_numeric(
        dataframe["poi_id"],
        errors="raise",
    ).astype(int)

    if len(dataframe) != EXPECTED_POI_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_POI_COUNT} POIs, "
            f"but found {len(dataframe)}."
        )

    if dataframe["poi_id"].duplicated().any():
        duplicate_ids = dataframe.loc[
            dataframe["poi_id"].duplicated(keep=False),
            ["poi_id", "name"],
        ]

        raise ValueError(
            "Duplicate POI IDs were found:\n"
            f"{duplicate_ids.to_string(index=False)}"
        )

    expected_ids = list(range(1, EXPECTED_POI_COUNT + 1))
    actual_ids = sorted(dataframe["poi_id"].tolist())

    if actual_ids != expected_ids:
        missing_ids = sorted(set(expected_ids) - set(actual_ids))
        unexpected_ids = sorted(set(actual_ids) - set(expected_ids))

        raise ValueError(
            "POI IDs must be exactly 1 to 50.\n"
            f"Missing IDs: {missing_ids}\n"
            f"Unexpected IDs: {unexpected_ids}"
        )

    # ------------------------------------------------------------
    # Name validation
    # ------------------------------------------------------------

    if dataframe["name"].isna().any():
        raise ValueError("One or more POI names are missing.")

    normalised_names = (
        dataframe["name"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
    )

    if normalised_names.eq("").any():
        raise ValueError("One or more POI names are blank.")

    if normalised_names.duplicated().any():
        duplicate_names = dataframe.loc[
            normalised_names.duplicated(keep=False),
            ["poi_id", "name"],
        ]

        raise ValueError(
            "Duplicate POI names were found:\n"
            f"{duplicate_names.to_string(index=False)}"
        )

    # ------------------------------------------------------------
    # Numeric conversion
    # ------------------------------------------------------------

    numeric_columns = [
        "latitude",
        "longitude",
        "google_rating",
        "review_count",
        "visit_duration_min",
        "admission_cost_rm",
        "indoor_ratio",
        "shelter_ratio",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    # ------------------------------------------------------------
    # Numeric range validation
    # ------------------------------------------------------------

    if not dataframe["latitude"].between(-90, 90).all():
        raise ValueError("Latitude values must be between -90 and 90.")

    if not dataframe["longitude"].between(-180, 180).all():
        raise ValueError(
            "Longitude values must be between -180 and 180."
        )

    if not dataframe["google_rating"].between(0, 5).all():
        raise ValueError(
            "Google ratings must be between 0 and 5."
        )

    if (dataframe["review_count"] < 0).any():
        raise ValueError("Review counts cannot be negative.")

    if (dataframe["visit_duration_min"] <= 0).any():
        raise ValueError(
            "Visit durations must be greater than zero."
        )

    if (dataframe["admission_cost_rm"] < 0).any():
        raise ValueError(
            "Admission costs cannot be negative."
        )

    if not dataframe["indoor_ratio"].between(0, 1).all():
        raise ValueError(
            "Indoor ratios must be between 0 and 1."
        )

    if not dataframe["shelter_ratio"].between(0, 1).all():
        raise ValueError(
            "Shelter ratios must be between 0 and 1."
        )

    # ------------------------------------------------------------
    # Zone validation
    # ------------------------------------------------------------

    dataframe["zone"] = (
        dataframe["zone"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    invalid_zones = sorted(
        set(dataframe["zone"]) - set(ZONE_SCORES)
    )

    if invalid_zones:
        raise ValueError(
            f"Invalid zones found: {invalid_zones}. "
            "Only A, B and C are allowed."
        )

    # ------------------------------------------------------------
    # Coordinate duplication
    # ------------------------------------------------------------

    duplicate_coordinates = dataframe.duplicated(
        subset=["latitude", "longitude"],
        keep=False,
    )

    if duplicate_coordinates.any():
        duplicated = dataframe.loc[
            duplicate_coordinates,
            ["poi_id", "name", "latitude", "longitude"],
        ]

        raise ValueError(
            "Multiple POIs share identical coordinates:\n"
            f"{duplicated.to_string(index=False)}"
        )

    return (
        dataframe
        .sort_values("poi_id")
        .reset_index(drop=True)
    )


# ================================================================
# Utility calculation
# ================================================================

def calculate_utility(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the utility components described in Chapter 3.

    Rating component:
        R_i = 2 × GoogleRating_i

    Review component:
        C_i =
        10 × log10(ReviewCount_i + 1)
        / log10(MaxReviewCount + 1)

    Zone component:
        P_i = 10 for Zone A
        P_i = 8 for Zone B
        P_i = 6 for Zone C

    Final utility:
        U_i = (R_i + C_i + P_i) / 3
    """

    result = dataframe.copy()

    # Rating is converted from the original 0–5 scale to 0–10.
    result["rating_norm"] = (
        2.0 * result["google_rating"]
    )

    maximum_review_count = result["review_count"].max()

    if maximum_review_count <= 0:
        raise ValueError(
            "At least one POI must have a review count above zero."
        )

    result["review_norm"] = (
        10.0
        * np.log10(result["review_count"] + 1)
        / np.log10(maximum_review_count + 1)
    )

    result["zone_score"] = result["zone"].map(ZONE_SCORES)

    result["utility_score"] = (
        result["rating_norm"]
        + result["review_norm"]
        + result["zone_score"]
    ) / 3.0

    result["utility_formula_version"] = (
        UTILITY_FORMULA_VERSION
    )

    calculated_columns = [
        "rating_norm",
        "review_norm",
        "zone_score",
        "utility_score",
    ]

    result[calculated_columns] = result[
        calculated_columns
    ].round(UTILITY_DECIMAL_PLACES)

    return result


# ================================================================
# Walking-time calculation
# ================================================================

def calculate_equirectangular_distance_metres(
    latitude_i: float,
    longitude_i: float,
    latitude_j: float,
    longitude_j: float,
) -> float:
    """
    Calculate straight-line distance with an equirectangular
    approximation.

    x = Δlongitude × cos(mean latitude)
    y = Δlatitude

    Distance:
        d_ij = R × sqrt(x² + y²)

    Latitude and longitude are converted to radians.
    """

    latitude_i_rad = math.radians(latitude_i)
    latitude_j_rad = math.radians(latitude_j)

    longitude_i_rad = math.radians(longitude_i)
    longitude_j_rad = math.radians(longitude_j)

    mean_latitude_rad = (
        latitude_i_rad + latitude_j_rad
    ) / 2.0

    x = (
        longitude_j_rad - longitude_i_rad
    ) * math.cos(mean_latitude_rad)

    y = latitude_j_rad - latitude_i_rad

    return EARTH_RADIUS_METRES * math.sqrt(
        x ** 2 + y ** 2
    )


def generate_walking_time_matrix(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate a symmetric walking-time matrix in minutes.

    Walking time:
        T_ij = distance_ij / (1.4 × 60)

    The matrix is a coordinate-derived proxy, not a
    pedestrian street-network route.
    """

    poi_ids = dataframe["poi_id"].tolist()
    poi_count = len(dataframe)

    matrix = np.zeros(
        shape=(poi_count, poi_count),
        dtype=float,
    )

    for i in range(poi_count):
        for j in range(i + 1, poi_count):

            distance_metres = (
                calculate_equirectangular_distance_metres(
                    latitude_i=dataframe.loc[i, "latitude"],
                    longitude_i=dataframe.loc[i, "longitude"],
                    latitude_j=dataframe.loc[j, "latitude"],
                    longitude_j=dataframe.loc[j, "longitude"],
                )
            )

            walking_minutes = (
                distance_metres
                / WALKING_SPEED_METRES_PER_SECOND
                / 60.0
            )

            walking_minutes = round(
                walking_minutes,
                MATRIX_DECIMAL_PLACES,
            )

            # Mirror the value to guarantee exact symmetry.
            matrix[i, j] = walking_minutes
            matrix[j, i] = walking_minutes

    np.fill_diagonal(matrix, 0.0)

    matrix_dataframe = pd.DataFrame(
        matrix,
        columns=[str(poi_id) for poi_id in poi_ids],
    )

    matrix_dataframe.insert(
        0,
        "poi_id",
        poi_ids,
    )

    return matrix_dataframe


# ================================================================
# Output validation
# ================================================================

def validate_calculated_dataset(
    dataframe: pd.DataFrame,
) -> dict[str, object]:
    """Validate all utility-related calculations."""

    rating_expected = (
        2.0 * dataframe["google_rating"]
    )

    maximum_review_count = dataframe["review_count"].max()

    review_expected = (
        10.0
        * np.log10(dataframe["review_count"] + 1)
        / np.log10(maximum_review_count + 1)
    )

    zone_expected = dataframe["zone"].map(ZONE_SCORES)

    utility_expected = (
        rating_expected
        + review_expected
        + zone_expected
    ) / 3.0

    tolerance = 10 ** (-UTILITY_DECIMAL_PLACES)

    checks = {
        "row_count": len(dataframe),
        "poi_ids_exactly_1_to_50": (
            dataframe["poi_id"].tolist()
            == list(range(1, EXPECTED_POI_COUNT + 1))
        ),
        "rating_norm_mismatch_count": int(
            (
                np.abs(
                    dataframe["rating_norm"]
                    - rating_expected
                )
                > tolerance
            ).sum()
        ),
        "review_norm_mismatch_count": int(
            (
                np.abs(
                    dataframe["review_norm"]
                    - review_expected
                )
                > tolerance
            ).sum()
        ),
        "zone_score_mismatch_count": int(
            (
                np.abs(
                    dataframe["zone_score"]
                    - zone_expected
                )
                > tolerance
            ).sum()
        ),
        "utility_score_mismatch_count": int(
            (
                np.abs(
                    dataframe["utility_score"]
                    - utility_expected
                )
                > tolerance
            ).sum()
        ),
        "minimum_utility": float(
            dataframe["utility_score"].min()
        ),
        "maximum_utility": float(
            dataframe["utility_score"].max()
        ),
    }

    checks["passed"] = all(
        [
            checks["row_count"] == EXPECTED_POI_COUNT,
            checks["poi_ids_exactly_1_to_50"],
            checks["rating_norm_mismatch_count"] == 0,
            checks["review_norm_mismatch_count"] == 0,
            checks["zone_score_mismatch_count"] == 0,
            checks["utility_score_mismatch_count"] == 0,
        ]
    )

    return checks


def validate_walking_matrix(
    matrix_dataframe: pd.DataFrame,
) -> dict[str, object]:
    """Validate structure and values of the walking matrix."""

    row_ids = matrix_dataframe["poi_id"].tolist()

    matrix = matrix_dataframe.drop(
        columns=["poi_id"]
    ).to_numpy(dtype=float)

    expected_shape = (
        EXPECTED_POI_COUNT,
        EXPECTED_POI_COUNT,
    )

    diagonal = np.diag(matrix)

    non_diagonal_mask = ~np.eye(
        EXPECTED_POI_COUNT,
        dtype=bool,
    )

    non_diagonal_values = matrix[
        non_diagonal_mask
    ]

    maximum_asymmetry = float(
        np.max(np.abs(matrix - matrix.T))
    )

    checks = {
        "matrix_shape": list(matrix.shape),
        "expected_shape": list(expected_shape),
        "shape_valid": matrix.shape == expected_shape,
        "row_ids_exactly_1_to_50": (
            row_ids
            == list(range(1, EXPECTED_POI_COUNT + 1))
        ),
        "column_ids_exactly_1_to_50": (
            list(matrix_dataframe.columns[1:])
            == [
                str(value)
                for value in range(
                    1,
                    EXPECTED_POI_COUNT + 1,
                )
            ]
        ),
        "finite_values": bool(
            np.isfinite(matrix).all()
        ),
        "negative_value_count": int(
            (matrix < 0).sum()
        ),
        "zero_diagonal": bool(
            np.allclose(
                diagonal,
                0.0,
                atol=1e-12,
            )
        ),
        "zero_non_diagonal_count": int(
            np.isclose(
                non_diagonal_values,
                0.0,
                atol=1e-12,
            ).sum()
        ),
        "positive_non_diagonal": bool(
            np.all(non_diagonal_values > 0)
        ),
        "symmetric": bool(
            np.allclose(
                matrix,
                matrix.T,
                atol=1e-12,
            )
        ),
        "maximum_asymmetry_minutes": (
            maximum_asymmetry
        ),
        "minimum_non_diagonal_minutes": float(
            non_diagonal_values.min()
        ),
        "maximum_non_diagonal_minutes": float(
            non_diagonal_values.max()
        ),
    }

    checks["passed"] = all(
        [
            checks["shape_valid"],
            checks["row_ids_exactly_1_to_50"],
            checks["column_ids_exactly_1_to_50"],
            checks["finite_values"],
            checks["negative_value_count"] == 0,
            checks["zero_diagonal"],
            checks["zero_non_diagonal_count"] == 0,
            checks["positive_non_diagonal"],
            checks["symmetric"],
        ]
    )

    return checks


# ================================================================
# Main execution
# ================================================================

def main() -> None:
    # Load manually collected values.
    raw_poi_dataframe = load_poi_dataset(
        INPUT_POI_FILE
    )

    # Calculate all utility-related fields.
    calculated_poi_dataframe = calculate_utility(
        raw_poi_dataframe
    )

    # Generate the walking-time matrix.
    walking_matrix_dataframe = (
        generate_walking_time_matrix(
            calculated_poi_dataframe
        )
    )

    # Validate outputs before writing final files.
    dataset_validation = (
        validate_calculated_dataset(
            calculated_poi_dataframe
        )
    )

    matrix_validation = validate_walking_matrix(
        walking_matrix_dataframe
    )

    if not dataset_validation["passed"]:
        raise ValueError(
            "Calculated dataset validation failed:\n"
            f"{dataset_validation}"
        )

    if not matrix_validation["passed"]:
        raise ValueError(
            "Walking matrix validation failed:\n"
            f"{matrix_validation}"
        )

    # Save files only after successful validation.
    calculated_poi_dataframe.to_csv(
        OUTPUT_POI_FILE,
        index=False,
        float_format=f"%.{UTILITY_DECIMAL_PLACES}f",
    )

    walking_matrix_dataframe.to_csv(
        OUTPUT_MATRIX_FILE,
        index=False,
        float_format=f"%.{MATRIX_DECIMAL_PLACES}f",
    )

    validation_report = {
        "input_file": INPUT_POI_FILE.name,
        "calculated_poi_file": OUTPUT_POI_FILE.name,
        "walking_matrix_file": OUTPUT_MATRIX_FILE.name,
        "utility_formula": {
            "rating_norm": (
                "2 * google_rating"
            ),
            "review_norm": (
                "10 * log10(review_count + 1) "
                "/ log10(max_review_count + 1)"
            ),
            "zone_score": ZONE_SCORES,
            "utility_score": (
                "(rating_norm + review_norm "
                "+ zone_score) / 3"
            ),
            "formula_version": (
                UTILITY_FORMULA_VERSION
            ),
        },
        "walking_formula": {
            "method": (
                "Equirectangular straight-line "
                "distance approximation"
            ),
            "earth_radius_metres": (
                EARTH_RADIUS_METRES
            ),
            "walking_speed_metres_per_second": (
                WALKING_SPEED_METRES_PER_SECOND
            ),
            "walking_time": (
                "distance_metres / 1.4 / 60"
            ),
        },
        "dataset_validation": dataset_validation,
        "matrix_validation": matrix_validation,
    }

    # Add checksums after the output files exist.
    validation_report["sha256"] = {
        "input_poi_file": calculate_sha256(
            INPUT_POI_FILE
        ),
        "calculated_poi_file": calculate_sha256(
            OUTPUT_POI_FILE
        ),
        "walking_matrix_file": calculate_sha256(
            OUTPUT_MATRIX_FILE
        ),
    }

    OUTPUT_VALIDATION_FILE.write_text(
        json.dumps(
            validation_report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 65)
    print("TRIPMATES LITE DATA PREPARATION COMPLETED")
    print("=" * 65)

    print(
        f"Input dataset: "
        f"{INPUT_POI_FILE.resolve()}"
    )

    print(
        f"Calculated dataset: "
        f"{OUTPUT_POI_FILE.resolve()}"
    )

    print(
        f"Walking matrix: "
        f"{OUTPUT_MATRIX_FILE.resolve()}"
    )

    print(
        f"Validation report: "
        f"{OUTPUT_VALIDATION_FILE.resolve()}"
    )

    print()
    print(
        "Dataset validation passed:",
        dataset_validation["passed"],
    )

    print(
        "Matrix validation passed:",
        matrix_validation["passed"],
    )

    print(
        "Utility range:",
        f"{dataset_validation['minimum_utility']:.6f}",
        "to",
        f"{dataset_validation['maximum_utility']:.6f}",
    )

    print(
        "Walking-time range:",
        f"{matrix_validation['minimum_non_diagonal_minutes']:.6f}",
        "to",
        f"{matrix_validation['maximum_non_diagonal_minutes']:.6f}",
        "minutes",
    )

    print()
    print("SHA-256 hashes:")

    for file_name, file_hash in (
        validation_report["sha256"].items()
    ):
        print(f"  {file_name}: {file_hash}")


if __name__ == "__main__":
    main()