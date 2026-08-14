#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RSRL_HU.py

Robust state residual learning informed by hydrology with prequential
uncertainty quantification for one-day-ahead landslide velocity forecasting.

This single-file research implementation provides the principal forecasting
workflow described in the associated manuscript. It is intentionally limited
to the proposed RSRL-HU framework and its primary temporal evaluation. It does
not reproduce benchmark-model comparisons, graphical outputs, bootstrap
experiments, permutation analyses, ablation studies, or supplementary
sensitivity experiments.

Principal workflow
------------------
1. Validate the complete daily monitoring record.
2. Estimate causal latent displacement and velocity states from cumulative
   GNSS displacement using Student-t innovation weighting.
3. Construct causal motion-state, state-estimation, rainfall, and groundwater
   predictors.
4. Forecast following-day groundwater-depth change with a Huber gradient
   boosting regressor.
5. Generate temporally external groundwater predictions through forward
   cross-fitting.
6. Select groundwater persistence shrinkage from temporally external
   development predictions.
7. estimate a bounded hydrology-informed dynamic velocity prior.
8. Learn shrinkage-controlled residual corrections using compact Huber
   gradient boosting regressors.
9. Select residual-model complexity through nested temporal validation.
10. Evaluate four rolling-origin blocks and one prespecified late-period
    validation block.
11. Construct rolling prequential empirical prediction intervals from
    sequentially available absolute forecast errors.
12. Generate an optional operational forecast for the day following the final
    monitoring observation.

Chronological information boundary
----------------------------------
At forecast origin t, every predictor is constructed exclusively from rainfall,
groundwater-depth, and cumulative-displacement observations available at or
before day t. Observations from day t+1 are used only after the forecast has
been generated, for reference-target construction, external evaluation, and
prequential error updating.

Input file
----------
The default input file is:

    data.xlsx

The workbook must be located in the same directory as this script and must
contain the following columns:

    Date
    RL (mm/d)
    GL (m)
    SD (mm)

Command-line use
----------------
Run the complete primary workflow:

    python RSRL_HU.py

Specify an alternative workbook:

    python RSRL_HU.py --data /path/to/data.xlsx

Omit the operational forecast:

    python RSRL_HU.py --skip-operational

Output behavior
---------------
All results are printed to the terminal. The script does not create figures,
models, tables, spreadsheets, or any other output files.
"""

from __future__ import annotations

import argparse
import platform
import random
import sys
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


# =============================================================================
# Global configuration
# =============================================================================

MODEL_NAME = "RSRL-HU"
MODEL_VERSION = "1.0.0"

RANDOM_SEED = 42

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = SCRIPT_DIRECTORY / "data.xlsx"

REQUIRED_COLUMNS = [
    "Date",
    "RL (mm/d)",
    "GL (m)",
    "SD (mm)",
]

EXPECTED_RECORD_START = pd.Timestamp("2016-01-12")
EXPECTED_RECORD_END = pd.Timestamp("2017-12-31")
EXPECTED_RECORD_LENGTH = 720

# The first eligible target date is 2 February 2016. Its zero-based raw-row
# index is 21 in the complete 720-day monitoring record.
SAMPLE_START_ROW = 21

# Nested temporal validation.
INNER_SPLITS = 3
INNER_TEST_SIZE = 35
INNER_GAP = 1

# Groundwater forward cross-fitting.
HYDRO_WARMUP = 90
HYDRO_BLOCK_SIZE = 30

HYDRO_SHRINKAGE_GRID = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]

# Residual shrinkage.
RESIDUAL_SHRINKAGE_GRID = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
]

# Rolling prequential uncertainty quantification.
INTERVAL_ALPHA = 0.10
ERROR_MEMORY_WINDOW = 120

# Hydrological activation depth.
GROUNDWATER_ACTIVATION_DEPTH = 12.30

FILTER_CONFIG = {
    "phi": 0.90,
    "level_process_ratio": 0.25,
    "velocity_process_ratio": 0.08,
    "minimum_observation_scale": 0.05,
    "student_degrees_of_freedom": 4.0,
    "minimum_observation_weight": 0.03,
    "maximum_standardized_innovation": 10.0,
    "initial_velocity": 0.0,
}

HYDRO_PARAMETERS = {
    "n_estimators": 100,
    "learning_rate": 0.03,
    "max_depth": 2,
    "min_samples_leaf": 20,
}

RESIDUAL_CONFIGURATIONS = [
    {
        "configuration": "C1",
        "n_estimators": 50,
        "learning_rate": 0.03,
        "max_depth": 1,
        "min_samples_leaf": 20,
    },
    {
        "configuration": "C2",
        "n_estimators": 100,
        "learning_rate": 0.03,
        "max_depth": 1,
        "min_samples_leaf": 20,
    },
    {
        "configuration": "C3",
        "n_estimators": 100,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_samples_leaf": 20,
    },
    {
        "configuration": "C4",
        "n_estimators": 150,
        "learning_rate": 0.02,
        "max_depth": 2,
        "min_samples_leaf": 20,
    },
]

PRIOR_INITIAL_PARAMETERS = np.array(
    [0.90, 0.00, 0.01],
    dtype=float,
)

PRIOR_LOWER_BOUNDS = np.array(
    [0.00, -0.05, 0.00],
    dtype=float,
)

PRIOR_UPPER_BOUNDS = np.array(
    [0.999, 0.05, 0.20],
    dtype=float,
)


# =============================================================================
# Predictor definitions
# =============================================================================

CORE_STATE_FEATURES = [
    "latent_velocity",
    "velocity_lag1",
    "velocity_lag2",
    "velocity_lag6",
    "velocity_mean3",
    "velocity_mean7",
    "velocity_std7",
    "velocity_max7",
    "velocity_min7",
    "latent_dSD",
    "latent_acceleration",
    "prior_velocity",
]

STATE_UNCERTAINTY_FEATURES = [
    "innovation",
    "standardized_innovation",
    "observation_weight",
    "outlier_score",
    "level_variance",
    "velocity_variance",
]

GROUNDWATER_HISTORY_FEATURES = [
    "GL",
    "GWR",
    "GL_mean3",
    "GL_mean7",
    "GL_mean14",
    "GL_std7",
    "GL_min7",
    "GL_max7",
    "GWR_sum3",
    "GWR_sum7",
    "GWR_sum14",
]

RAINFALL_HISTORY_FEATURES = [
    "RL",
    "RL_sum3",
    "RL_sum7",
    "RL_sum14",
    "RL_sum21",
    "RL_max3",
    "RL_max7",
    "wet_days7",
    "wet_days14",
    "API3",
    "API7",
    "API14",
    "DOY_sin",
    "DOY_cos",
]

HYDROLOGICAL_FEATURES = (
    GROUNDWATER_HISTORY_FEATURES
    + RAINFALL_HISTORY_FEATURES
)

GROUNDWATER_FORECAST_FEATURES = [
    "Predicted_GL_next",
    "Predicted_GWR_next",
    "Effective_head",
    "Pressure_scaled",
    "GL_anomaly7",
]

RESIDUAL_FEATURES = (
    ["Hydrology_prior"]
    + CORE_STATE_FEATURES
    + STATE_UNCERTAINTY_FEATURES
    + GROUNDWATER_HISTORY_FEATURES
    + GROUNDWATER_FORECAST_FEATURES
)


# =============================================================================
# General utilities
# =============================================================================

def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """Set deterministic Python and NumPy random seeds."""

    random.seed(seed)
    np.random.seed(seed)


def require_finite_array(
    values: np.ndarray | pd.Series | list[float],
    name: str,
    expected_length: int | None = None,
) -> np.ndarray:
    """Convert an array-like object to a finite one-dimensional float array."""

    array = np.asarray(values, dtype=float).reshape(-1)

    if expected_length is not None and len(array) != expected_length:
        raise RuntimeError(
            f"{name} contains {len(array)} values, whereas "
            f"{expected_length} values were expected."
        )

    if not np.all(np.isfinite(array)):
        raise RuntimeError(
            f"{name} contains one or more non-finite values."
        )

    return array


def robust_scale(
    values: np.ndarray | pd.Series,
    minimum: float = 1.0e-8,
) -> float:
    """Estimate scale using the normalized median absolute deviation."""

    array = np.asarray(values, dtype=float).reshape(-1)
    array = array[np.isfinite(array)]

    if len(array) == 0:
        return float(minimum)

    median = float(np.median(array))
    scale = 1.4826 * float(
        np.median(np.abs(array - median))
    )

    if not np.isfinite(scale) or scale < minimum:
        scale = float(np.std(array))

    if not np.isfinite(scale) or scale < minimum:
        scale = float(minimum)

    return float(scale)


def root_mean_squared_error(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Calculate root mean squared error."""

    return float(
        np.sqrt(
            mean_squared_error(
                reference,
                prediction,
            )
        )
    )


def point_metrics(
    reference: np.ndarray | pd.Series,
    prediction: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Calculate deterministic point-forecast metrics."""

    reference_array = np.asarray(
        reference,
        dtype=float,
    )

    prediction_array = np.asarray(
        prediction,
        dtype=float,
    )

    if reference_array.shape != prediction_array.shape:
        raise ValueError(
            "Reference and prediction arrays have incompatible shapes: "
            f"{reference_array.shape} and {prediction_array.shape}."
        )

    valid = (
        np.isfinite(reference_array)
        & np.isfinite(prediction_array)
    )

    reference_array = reference_array[valid]
    prediction_array = prediction_array[valid]

    if len(reference_array) == 0:
        return {
            "N": 0,
            "RMSE": np.nan,
            "MAE": np.nan,
            "R2": np.nan,
            "Bias": np.nan,
        }

    signed_error = (
        prediction_array - reference_array
    )

    if (
        len(reference_array) >= 2
        and np.std(reference_array) > 1.0e-12
    ):
        coefficient_of_determination = float(
            r2_score(
                reference_array,
                prediction_array,
            )
        )
    else:
        coefficient_of_determination = np.nan

    return {
        "N": int(len(reference_array)),
        "RMSE": root_mean_squared_error(
            reference_array,
            prediction_array,
        ),
        "MAE": float(
            mean_absolute_error(
                reference_array,
                prediction_array,
            )
        ),
        "R2": coefficient_of_determination,
        "Bias": float(np.mean(signed_error)),
    }


def temporal_selection_score(
    reference: np.ndarray | pd.Series,
    prediction: np.ndarray | pd.Series,
) -> float:
    """
    Calculate the temporal model-selection score.

    Score = MAE + 0.5 * RMSE + 0.05 * absolute bias.
    """

    metrics = point_metrics(
        reference,
        prediction,
    )

    if metrics["N"] == 0:
        return np.inf

    return float(
        metrics["MAE"]
        + 0.50 * metrics["RMSE"]
        + 0.05 * abs(metrics["Bias"])
    )


def percentage_skill(
    candidate_error: float,
    reference_error: float,
) -> float:
    """Calculate percentage skill relative to a reference error."""

    if (
        not np.isfinite(candidate_error)
        or not np.isfinite(reference_error)
        or abs(reference_error) <= 1.0e-12
    ):
        return np.nan

    return float(
        100.0
        * (
            1.0
            - candidate_error / reference_error
        )
    )


def finite_sample_error_quantile(
    errors: np.ndarray | list[float],
    alpha: float = INTERVAL_ALPHA,
) -> float:
    """
    Calculate the finite-sample-corrected empirical error quantile.

    The selected one-indexed order statistic is

        ceil((n + 1) * (1 - alpha)),

    capped at n.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError(
            "The interval miscoverage rate must lie strictly between zero "
            "and one."
        )

    array = np.asarray(errors, dtype=float).reshape(-1)
    array = array[np.isfinite(array)]

    if len(array) == 0:
        raise ValueError(
            "At least one finite error is required."
        )

    ordered = np.sort(array)

    rank = int(
        np.ceil(
            (len(ordered) + 1)
            * (1.0 - alpha)
        )
    )

    rank = int(
        np.clip(
            rank,
            1,
            len(ordered),
        )
    )

    return float(ordered[rank - 1])


def prediction_interval_metrics(
    reference: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = INTERVAL_ALPHA,
) -> dict[str, float]:
    """Calculate empirical prediction-interval metrics."""

    if not 0.0 < alpha < 1.0:
        raise ValueError(
            "The interval miscoverage rate must lie strictly between zero "
            "and one."
        )

    reference = np.asarray(reference, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    if not (
        reference.shape
        == lower.shape
        == upper.shape
    ):
        raise ValueError(
            "Reference, lower-bound, and upper-bound arrays must have "
            "identical shapes."
        )

    valid = (
        np.isfinite(reference)
        & np.isfinite(lower)
        & np.isfinite(upper)
    )

    reference = reference[valid]
    lower = lower[valid]
    upper = upper[valid]

    if len(reference) == 0:
        return {
            "Covered": 0,
            "PICP": np.nan,
            "MPIW": np.nan,
            "Median_width": np.nan,
            "Winkler": np.nan,
        }

    if np.any(lower > upper):
        raise RuntimeError(
            "At least one prediction-interval lower bound exceeds its "
            "corresponding upper bound."
        )

    width = upper - lower

    covered = (
        (reference >= lower)
        & (reference <= upper)
    )

    winkler = width.copy()

    below = reference < lower
    above = reference > upper

    winkler[below] += (
        2.0
        / alpha
        * (lower[below] - reference[below])
    )

    winkler[above] += (
        2.0
        / alpha
        * (reference[above] - upper[above])
    )

    return {
        "Covered": int(np.sum(covered)),
        "PICP": float(np.mean(covered)),
        "MPIW": float(np.mean(width)),
        "Median_width": float(np.median(width)),
        "Winkler": float(np.mean(winkler)),
    }


def calculate_antecedent_precipitation_index(
    rainfall: np.ndarray | pd.Series,
    decay_scale: float,
) -> np.ndarray:
    """Calculate a causal antecedent precipitation index."""

    if decay_scale <= 0.0:
        raise ValueError(
            "The antecedent-precipitation decay scale must be positive."
        )

    rainfall_array = require_finite_array(
        rainfall,
        "Rainfall observations",
    )

    result = np.zeros(
        len(rainfall_array),
        dtype=float,
    )

    decay = np.exp(
        -1.0 / float(decay_scale)
    )

    for position in range(len(rainfall_array)):
        if position == 0:
            result[position] = rainfall_array[position]
        else:
            result[position] = (
                rainfall_array[position]
                + decay * result[position - 1]
            )

    return result


def make_huber_gradient_boosting(
    parameters: dict[str, Any],
) -> GradientBoostingRegressor:
    """Construct a deterministic Huber gradient boosting regressor."""

    return GradientBoostingRegressor(
        loss="huber",
        alpha=0.90,
        n_estimators=int(
            parameters["n_estimators"]
        ),
        learning_rate=float(
            parameters["learning_rate"]
        ),
        max_depth=int(
            parameters["max_depth"]
        ),
        min_samples_leaf=int(
            parameters["min_samples_leaf"]
        ),
        subsample=0.85,
        random_state=RANDOM_SEED,
    )


def installed_package_version(
    package_name: str,
) -> str:
    """Return the installed version of a package."""

    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not available"


# =============================================================================
# Data validation
# =============================================================================

def resolve_data_file(
    requested_path: Path | None,
) -> Path:
    """Resolve and validate the input workbook path."""

    candidate = (
        requested_path.expanduser().resolve()
        if requested_path is not None
        else DEFAULT_DATA_FILE.resolve()
    )

    if not candidate.exists():
        raise FileNotFoundError(
            f"Monitoring-data workbook not found: {candidate}"
        )

    if not candidate.is_file():
        raise FileNotFoundError(
            f"The selected data path is not a regular file: {candidate}"
        )

    if candidate.suffix.lower() != ".xlsx":
        raise ValueError(
            "The monitoring-data file must be an Excel workbook with the "
            ".xlsx extension."
        )

    return candidate


def read_monitoring_data(
    file_path: Path,
) -> pd.DataFrame:
    """Read and comprehensively validate the monitoring record."""

    data = pd.read_excel(
        file_path,
        engine="openpyxl",
    )

    data.columns = (
        data.columns
        .astype(str)
        .str.replace("\u3000", " ", regex=False)
        .str.replace("\xa0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "The workbook does not contain all required columns. "
            f"Missing columns: {missing_columns}. "
            f"Detected columns: {data.columns.tolist()}."
        )

    data = (
        data[REQUIRED_COLUMNS]
        .rename(
            columns={
                "RL (mm/d)": "RL",
                "GL (m)": "GL",
                "SD (mm)": "SD",
            }
        )
        .copy()
    )

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce",
    )

    for column in ["RL", "GL", "SD"]:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = (
        data.sort_values("Date")
        .reset_index(drop=True)
    )

    if data.isna().any().any():
        missing_summary = data.isna().sum()
        missing_summary = missing_summary[
            missing_summary > 0
        ]

        raise ValueError(
            "Missing or invalid observations were detected:\n"
            f"{missing_summary.to_string()}"
        )

    if data["Date"].duplicated().any():
        duplicated_dates = data.loc[
            data["Date"].duplicated(keep=False),
            "Date",
        ].dt.strftime("%Y-%m-%d").tolist()

        raise ValueError(
            "Duplicated monitoring dates were detected: "
            f"{duplicated_dates}"
        )

    expected_dates = pd.date_range(
        start=data["Date"].min(),
        end=data["Date"].max(),
        freq="D",
    )

    observed_dates = pd.DatetimeIndex(
        data["Date"]
    )

    if not observed_dates.equals(expected_dates):
        missing_dates = expected_dates.difference(
            observed_dates
        )

        raise ValueError(
            "The monitoring record is not daily and continuous. "
            f"Missing dates include: {missing_dates[:20].tolist()}."
        )

    if len(data) != EXPECTED_RECORD_LENGTH:
        raise ValueError(
            "The manuscript dataset must contain exactly "
            f"{EXPECTED_RECORD_LENGTH} consecutive daily observations, "
            f"whereas {len(data)} observations were detected."
        )

    if data["Date"].iloc[0] != EXPECTED_RECORD_START:
        raise ValueError(
            "The manuscript dataset must begin on "
            f"{EXPECTED_RECORD_START.date()}, whereas the detected start "
            f"date is {data['Date'].iloc[0].date()}."
        )

    if data["Date"].iloc[-1] != EXPECTED_RECORD_END:
        raise ValueError(
            "The manuscript dataset must end on "
            f"{EXPECTED_RECORD_END.date()}, whereas the detected end date "
            f"is {data['Date'].iloc[-1].date()}."
        )

    if (data["RL"] < 0.0).any():
        raise ValueError(
            "Negative rainfall observations were detected."
        )

    for column in ["RL", "GL", "SD"]:
        values = data[column].to_numpy(dtype=float)

        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"Column {column} contains non-finite values."
            )

    if data["Date"].iloc[SAMPLE_START_ROW] != pd.Timestamp("2016-02-02"):
        raise RuntimeError(
            "The first eligible target row is inconsistent with the "
            "prespecified temporal design."
        )

    return data


# =============================================================================
# Student-t-weighted causal state estimation
# =============================================================================

def estimate_filter_parameters(
    displacement: np.ndarray,
    training_end: int,
    filter_config: dict[str, float],
) -> dict[str, float]:
    """
    Estimate observation and process scales from the development period.

    The training-end argument is an inclusive raw-row index. Only
    displacement observations available at or before that row are used.
    """

    displacement = require_finite_array(
        displacement,
        "Cumulative-displacement observations",
    )

    if training_end < 2:
        raise ValueError(
            "The filter development period is too short."
        )

    if training_end >= len(displacement):
        raise ValueError(
            "The filter training-end row lies outside the monitoring record."
        )

    development_displacement = displacement[
        :training_end + 1
    ]

    increment_scale = robust_scale(
        np.diff(development_displacement),
        minimum=filter_config[
            "minimum_observation_scale"
        ],
    )

    observation_scale = max(
        increment_scale / np.sqrt(2.0),
        filter_config[
            "minimum_observation_scale"
        ],
    )

    observation_variance = (
        observation_scale ** 2
    )

    level_process_variance = (
        filter_config["level_process_ratio"]
        * observation_scale
    ) ** 2

    velocity_process_variance = (
        filter_config["velocity_process_ratio"]
        * observation_scale
    ) ** 2

    return {
        "increment_scale": float(
            increment_scale
        ),
        "observation_scale": float(
            observation_scale
        ),
        "observation_variance": float(
            observation_variance
        ),
        "level_process_variance": float(
            level_process_variance
        ),
        "velocity_process_variance": float(
            velocity_process_variance
        ),
    }


def student_t_causal_state_filter(
    displacement: np.ndarray,
    parameters: dict[str, float],
    filter_config: dict[str, float],
) -> dict[str, np.ndarray]:
    """
    Run the Student-t-weighted causal displacement-velocity state estimator.

    The initial velocity is fixed at zero. It is not estimated from the
    second displacement observation because that observation is unavailable
    at the first filtering origin.
    """

    displacement = require_finite_array(
        displacement,
        "Cumulative-displacement observations",
    )

    number_of_observations = len(displacement)

    if number_of_observations < 2:
        raise ValueError(
            "At least two displacement observations are required."
        )

    phi = float(
        filter_config["phi"]
    )

    if not 0.0 <= phi <= 1.0:
        raise ValueError(
            "The velocity-persistence coefficient must lie in [0, 1]."
        )

    transition_matrix = np.array(
        [
            [1.0, 1.0],
            [0.0, phi],
        ],
        dtype=float,
    )

    observation_matrix = np.array(
        [[1.0, 0.0]],
        dtype=float,
    )

    process_covariance = np.diag(
        [
            parameters[
                "level_process_variance"
            ],
            parameters[
                "velocity_process_variance"
            ],
        ]
    )

    observation_variance = float(
        parameters["observation_variance"]
    )

    if observation_variance <= 0.0:
        raise ValueError(
            "The displacement observation variance must be positive."
        )

    state = np.array(
        [
            displacement[0],
            float(
                filter_config["initial_velocity"]
            ),
        ],
        dtype=float,
    )

    covariance = np.diag(
        [
            max(
                4.0 * observation_variance,
                0.01,
            ),
            max(
                parameters["increment_scale"] ** 2,
                0.01,
            ),
        ]
    )

    output_names = [
        "latent_SD",
        "latent_velocity",
        "prior_velocity",
        "innovation",
        "standardized_innovation",
        "observation_weight",
        "outlier_score",
        "level_variance",
        "velocity_variance",
    ]

    output = {
        name: np.zeros(
            number_of_observations,
            dtype=float,
        )
        for name in output_names
    }

    output["latent_SD"][0] = state[0]
    output["latent_velocity"][0] = state[1]
    output["prior_velocity"][0] = state[1]
    output["innovation"][0] = 0.0
    output["standardized_innovation"][0] = 0.0
    output["observation_weight"][0] = 1.0
    output["outlier_score"][0] = 0.0
    output["level_variance"][0] = covariance[0, 0]
    output["velocity_variance"][0] = covariance[1, 1]

    identity_matrix = np.eye(
        2,
        dtype=float,
    )

    degrees_of_freedom = float(
        filter_config[
            "student_degrees_of_freedom"
        ]
    )

    minimum_weight = float(
        filter_config[
            "minimum_observation_weight"
        ]
    )

    clipping_constant = float(
        filter_config[
            "maximum_standardized_innovation"
        ]
    )

    if degrees_of_freedom <= 0.0:
        raise ValueError(
            "The Student-t degrees of freedom must be positive."
        )

    if not 0.0 < minimum_weight <= 1.0:
        raise ValueError(
            "The minimum observation weight must lie in (0, 1]."
        )

    if clipping_constant <= 0.0:
        raise ValueError(
            "The innovation-clipping constant must be positive."
        )

    for position in range(
        1,
        number_of_observations,
    ):
        predicted_state = (
            transition_matrix @ state
        )

        predicted_covariance = (
            transition_matrix
            @ covariance
            @ transition_matrix.T
            + process_covariance
        )

        predicted_displacement = float(
            (
                observation_matrix
                @ predicted_state
            ).item()
        )

        innovation = float(
            displacement[position]
            - predicted_displacement
        )

        gaussian_innovation_variance = float(
            (
                observation_matrix
                @ predicted_covariance
                @ observation_matrix.T
            ).item()
            + observation_variance
        )

        gaussian_innovation_variance = max(
            gaussian_innovation_variance,
            1.0e-12,
        )

        standardized_innovation = float(
            innovation
            / np.sqrt(
                gaussian_innovation_variance
            )
        )

        observation_weight = (
            degrees_of_freedom + 1.0
        ) / (
            degrees_of_freedom
            + standardized_innovation ** 2
        )

        observation_weight = float(
            np.clip(
                observation_weight,
                minimum_weight,
                1.0,
            )
        )

        effective_observation_variance = (
            observation_variance
            / observation_weight
        )

        robust_innovation_variance = float(
            (
                observation_matrix
                @ predicted_covariance
                @ observation_matrix.T
            ).item()
            + effective_observation_variance
        )

        robust_innovation_variance = max(
            robust_innovation_variance,
            1.0e-12,
        )

        innovation_limit = (
            clipping_constant
            * np.sqrt(
                robust_innovation_variance
            )
        )

        clipped_innovation = float(
            np.clip(
                innovation,
                -innovation_limit,
                innovation_limit,
            )
        )

        kalman_gain = (
            predicted_covariance
            @ observation_matrix.T
            / robust_innovation_variance
        ).reshape(-1)

        state = (
            predicted_state
            + kalman_gain * clipped_innovation
        )

        gain_observation = np.outer(
            kalman_gain,
            observation_matrix.reshape(-1),
        )

        covariance = (
            (
                identity_matrix
                - gain_observation
            )
            @ predicted_covariance
            @ (
                identity_matrix
                - gain_observation
            ).T
            + np.outer(
                kalman_gain,
                kalman_gain,
            )
            * effective_observation_variance
        )

        covariance = 0.5 * (
            covariance + covariance.T
        )

        covariance[0, 0] = max(
            covariance[0, 0],
            0.0,
        )

        covariance[1, 1] = max(
            covariance[1, 1],
            0.0,
        )

        output["latent_SD"][position] = (
            state[0]
        )

        output["latent_velocity"][position] = (
            state[1]
        )

        output["prior_velocity"][position] = (
            predicted_state[1]
        )

        output["innovation"][position] = (
            innovation
        )

        output[
            "standardized_innovation"
        ][position] = standardized_innovation

        output[
            "observation_weight"
        ][position] = observation_weight

        output["outlier_score"][position] = (
            1.0 - observation_weight
        )

        output["level_variance"][position] = (
            covariance[0, 0]
        )

        output["velocity_variance"][position] = (
            covariance[1, 1]
        )

    for name, values in output.items():
        require_finite_array(
            values,
            f"State-filter output '{name}'",
            expected_length=number_of_observations,
        )

    return output


# =============================================================================
# Causal feature construction
# =============================================================================

def construct_causal_features(
    raw_subset: pd.DataFrame,
    filter_output: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Construct all causal state and hydrological predictors."""

    features = (
        raw_subset.copy()
        .reset_index(drop=True)
    )

    features["raw_row"] = np.arange(
        len(features),
        dtype=int,
    )

    for name, values in filter_output.items():
        features[name] = require_finite_array(
            values,
            f"State-filter feature '{name}'",
            expected_length=len(features),
        )

    features["latent_dSD"] = (
        features["latent_SD"].diff()
    )

    features["latent_acceleration"] = (
        features["latent_velocity"].diff()
    )

    features["velocity_lag1"] = (
        features["latent_velocity"].shift(1)
    )

    features["velocity_lag2"] = (
        features["latent_velocity"].shift(2)
    )

    features["velocity_lag6"] = (
        features["latent_velocity"].shift(6)
    )

    features["velocity_mean3"] = (
        features["latent_velocity"]
        .rolling(3)
        .mean()
    )

    features["velocity_mean7"] = (
        features["latent_velocity"]
        .rolling(7)
        .mean()
    )

    features["velocity_std7"] = (
        features["latent_velocity"]
        .rolling(7)
        .std()
    )

    features["velocity_max7"] = (
        features["latent_velocity"]
        .rolling(7)
        .max()
    )

    features["velocity_min7"] = (
        features["latent_velocity"]
        .rolling(7)
        .min()
    )

    # Groundwater depth is measured downward from the ground surface.
    # A positive GWR value therefore denotes a relative water-table rise.
    features["GWR"] = (
        -features["GL"].diff()
    )

    for window in [3, 7, 14]:
        features[f"GL_mean{window}"] = (
            features["GL"]
            .rolling(window)
            .mean()
        )

        features[f"GWR_sum{window}"] = (
            features["GWR"]
            .rolling(window)
            .sum()
        )

    features["GL_std7"] = (
        features["GL"]
        .rolling(7)
        .std()
    )

    features["GL_min7"] = (
        features["GL"]
        .rolling(7)
        .min()
    )

    features["GL_max7"] = (
        features["GL"]
        .rolling(7)
        .max()
    )

    for window in [3, 7, 14, 21]:
        features[f"RL_sum{window}"] = (
            features["RL"]
            .rolling(window)
            .sum()
        )

    features["RL_max3"] = (
        features["RL"]
        .rolling(3)
        .max()
    )

    features["RL_max7"] = (
        features["RL"]
        .rolling(7)
        .max()
    )

    features["wet_days7"] = (
        (features["RL"] > 0.1)
        .rolling(7)
        .sum()
    )

    features["wet_days14"] = (
        (features["RL"] > 0.1)
        .rolling(14)
        .sum()
    )

    for decay_scale in [3, 7, 14]:
        features[f"API{decay_scale}"] = (
            calculate_antecedent_precipitation_index(
                features["RL"],
                decay_scale,
            )
        )

    day_of_year = (
        features["Date"]
        .dt.dayofyear
        .to_numpy(dtype=float)
    )

    features["DOY_sin"] = np.sin(
        2.0
        * np.pi
        * day_of_year
        / 365.25
    )

    features["DOY_cos"] = np.cos(
        2.0
        * np.pi
        * day_of_year
        / 365.25
    )

    features["raw_increment"] = (
        features["SD"].diff()
    )

    return features


def construct_forecast_samples(
    feature_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct origin-target samples for one-day-ahead forecasting.

    Each row represents a forecast issued at origin t and evaluated against
    the reference latent velocity obtained after assimilating the cumulative
    displacement observation on day t+1.
    """

    samples = feature_data.copy()

    samples["target_row"] = (
        samples["raw_row"] + 1
    )

    samples["target_date"] = (
        samples["Date"].shift(-1)
    )

    samples["target_velocity"] = (
        samples["latent_velocity"].shift(-1)
    )

    samples["target_latent_SD"] = (
        samples["latent_SD"].shift(-1)
    )

    samples["target_observed_SD"] = (
        samples["SD"].shift(-1)
    )

    samples["target_GL"] = (
        samples["GL"].shift(-1)
    )

    samples["target_GL_delta"] = (
        samples["target_GL"]
        - samples["GL"]
    )

    required = (
        CORE_STATE_FEATURES
        + STATE_UNCERTAINTY_FEATURES
        + HYDROLOGICAL_FEATURES
        + [
            "target_date",
            "target_velocity",
            "target_latent_SD",
            "target_observed_SD",
            "target_GL",
            "target_GL_delta",
        ]
    )

    samples = samples.dropna(
        subset=required
    )

    samples = samples[
        samples["target_row"]
        >= SAMPLE_START_ROW
    ].copy()

    samples["target_row"] = (
        samples["target_row"]
        .astype(int)
    )

    samples = (
        samples
        .sort_values("target_row")
        .reset_index(drop=True)
    )

    if samples["target_row"].duplicated().any():
        raise RuntimeError(
            "Duplicated target rows were generated."
        )

    return samples


def prepare_dataset(
    raw: pd.DataFrame,
    filter_training_end: int,
    run_end: int,
    filter_config: dict[str, float],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, float],
]:
    """Prepare a fold-specific causal feature and forecast-sample dataset."""

    if filter_training_end < 0:
        raise ValueError(
            "The filter training-end row must be nonnegative."
        )

    if run_end < filter_training_end:
        raise ValueError(
            "The state-filter run end cannot precede the training end."
        )

    if run_end >= len(raw):
        raise ValueError(
            "The state-filter run end lies outside the monitoring record."
        )

    subset = (
        raw.iloc[:run_end + 1]
        .copy()
        .reset_index(drop=True)
    )

    parameters = estimate_filter_parameters(
        displacement=raw["SD"].to_numpy(
            dtype=float
        ),
        training_end=filter_training_end,
        filter_config=filter_config,
    )

    filter_output = student_t_causal_state_filter(
        displacement=subset["SD"].to_numpy(
            dtype=float
        ),
        parameters=parameters,
        filter_config=filter_config,
    )

    features = construct_causal_features(
        subset,
        filter_output,
    )

    samples = construct_forecast_samples(
        features
    )

    return samples, features, parameters


def select_target_rows(
    samples: pd.DataFrame,
    target_rows: np.ndarray,
) -> pd.DataFrame:
    """Select and validate a prespecified sequence of target rows."""

    requested = np.asarray(
        target_rows,
        dtype=int,
    ).reshape(-1)

    if len(requested) == 0:
        raise ValueError(
            "No target rows were requested."
        )

    if len(np.unique(requested)) != len(requested):
        raise ValueError(
            "The requested target-row sequence contains duplicates."
        )

    requested = np.sort(requested)

    selected = (
        samples[
            samples["target_row"].isin(requested)
        ]
        .sort_values("target_row")
        .reset_index(drop=True)
    )

    selected_rows = selected[
        "target_row"
    ].to_numpy(dtype=int)

    if not np.array_equal(
        selected_rows,
        requested,
    ):
        missing = sorted(
            set(requested)
            - set(selected_rows)
        )

        raise RuntimeError(
            "Forecast samples could not be constructed for target rows: "
            f"{missing[:20]}"
        )

    return selected


# =============================================================================
# Following-day groundwater forecasting
# =============================================================================

def fit_raw_groundwater_model(
    training_data: pd.DataFrame,
) -> GradientBoostingRegressor:
    """Fit the Huber regressor for following-day groundwater-depth change."""

    if len(training_data) < 2:
        raise ValueError(
            "At least two groundwater training samples are required."
        )

    target_change = require_finite_array(
        training_data["target_GL_delta"],
        "Groundwater training targets",
        expected_length=len(training_data),
    )

    model = make_huber_gradient_boosting(
        HYDRO_PARAMETERS
    )

    model.fit(
        training_data[
            HYDROLOGICAL_FEATURES
        ],
        target_change,
    )

    return model


def predict_raw_groundwater(
    model: GradientBoostingRegressor,
    data: pd.DataFrame,
) -> np.ndarray:
    """Predict following-day groundwater depth without persistence shrinkage."""

    predicted_change = model.predict(
        data[
            HYDROLOGICAL_FEATURES
        ]
    )

    predicted_change = require_finite_array(
        predicted_change,
        "Raw groundwater-depth-change predictions",
        expected_length=len(data),
    )

    current_groundwater = require_finite_array(
        data["GL"],
        "Current groundwater depth",
        expected_length=len(data),
    )

    predicted_groundwater = (
        current_groundwater
        + predicted_change
    )

    return require_finite_array(
        predicted_groundwater,
        "Raw following-day groundwater-depth predictions",
        expected_length=len(data),
    )


def forward_cross_fitted_groundwater(
    development_data: pd.DataFrame,
) -> np.ndarray:
    """
    Generate temporally external groundwater forecasts in consecutive blocks.

    Every block is predicted by a model fitted exclusively on development
    samples preceding that block.
    """

    predictions = np.full(
        len(development_data),
        np.nan,
        dtype=float,
    )

    if len(development_data) <= HYDRO_WARMUP:
        return predictions

    for block_start in range(
        HYDRO_WARMUP,
        len(development_data),
        HYDRO_BLOCK_SIZE,
    ):
        block_end = min(
            block_start + HYDRO_BLOCK_SIZE,
            len(development_data),
        )

        training = development_data.iloc[
            :block_start
        ]

        prediction_block = development_data.iloc[
            block_start:block_end
        ]

        model = fit_raw_groundwater_model(
            training
        )

        predictions[
            block_start:block_end
        ] = predict_raw_groundwater(
            model,
            prediction_block,
        )

    return predictions


def fit_groundwater_forecaster(
    development_data: pd.DataFrame,
) -> tuple[
    dict[str, Any],
    np.ndarray,
    np.ndarray,
]:
    """
    Fit the groundwater model and select persistence shrinkage.

    Shrinkage is selected using only temporally external development-period
    groundwater forecasts.
    """

    if len(development_data) <= HYDRO_WARMUP:
        raise RuntimeError(
            "The groundwater development period is shorter than the "
            "forward-cross-fitting warm-up."
        )

    raw_cross_fitted = (
        forward_cross_fitted_groundwater(
            development_data
        )
    )

    valid = np.isfinite(
        raw_cross_fitted
    )

    if int(np.sum(valid)) < 30:
        raise RuntimeError(
            "Insufficient temporally external groundwater predictions were "
            "available for shrinkage selection."
        )

    valid_positions = np.flatnonzero(
        valid
    )

    current_groundwater = require_finite_array(
        development_data.iloc[
            valid_positions
        ]["GL"],
        "Cross-fitted current groundwater depth",
    )

    reference_groundwater = require_finite_array(
        development_data.iloc[
            valid_positions
        ]["target_GL"],
        "Cross-fitted reference groundwater depth",
    )

    raw_valid_prediction = require_finite_array(
        raw_cross_fitted[valid],
        "Cross-fitted raw groundwater predictions",
        expected_length=len(current_groundwater),
    )

    search_records = []

    for shrinkage in HYDRO_SHRINKAGE_GRID:
        shrinkage = float(
            np.clip(
                shrinkage,
                0.0,
                1.0,
            )
        )

        prediction = (
            current_groundwater
            + shrinkage
            * (
                raw_valid_prediction
                - current_groundwater
            )
        )

        search_records.append(
            {
                "shrinkage": shrinkage,
                "score": temporal_selection_score(
                    reference_groundwater,
                    prediction,
                ),
            }
        )

    search_table = (
        pd.DataFrame(search_records)
        .sort_values(
            ["score", "shrinkage"],
            ascending=[True, True],
        )
        .reset_index(drop=True)
    )

    if len(search_table) == 0:
        raise RuntimeError(
            "Groundwater shrinkage selection produced no candidate result."
        )

    selected_shrinkage = float(
        np.clip(
            search_table.iloc[0][
                "shrinkage"
            ],
            0.0,
            1.0,
        )
    )

    shrunk_cross_fitted = np.full(
        len(development_data),
        np.nan,
        dtype=float,
    )

    shrunk_cross_fitted[valid] = (
        current_groundwater
        + selected_shrinkage
        * (
            raw_valid_prediction
            - current_groundwater
        )
    )

    final_model = fit_raw_groundwater_model(
        development_data
    )

    bundle = {
        "model": final_model,
        "shrinkage": selected_shrinkage,
        "search": search_table,
    }

    return (
        bundle,
        shrunk_cross_fitted,
        valid,
    )


def predict_groundwater(
    bundle: dict[str, Any],
    data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate persistence-shrunk and raw groundwater-depth predictions.

    Returns
    -------
    shrunk_prediction, raw_prediction
    """

    if "model" not in bundle:
        raise KeyError(
            "The groundwater bundle does not contain a fitted model."
        )

    if "shrinkage" not in bundle:
        raise KeyError(
            "The groundwater bundle does not contain a shrinkage coefficient."
        )

    raw_prediction = predict_raw_groundwater(
        bundle["model"],
        data,
    )

    current_groundwater = require_finite_array(
        data["GL"],
        "Current groundwater depth",
        expected_length=len(data),
    )

    shrinkage = float(
        np.clip(
            bundle["shrinkage"],
            0.0,
            1.0,
        )
    )

    shrunk_prediction = (
        current_groundwater
        + shrinkage
        * (
            raw_prediction
            - current_groundwater
        )
    )

    shrunk_prediction = require_finite_array(
        shrunk_prediction,
        "Persistence-shrunk groundwater-depth predictions",
        expected_length=len(data),
    )

    return (
        shrunk_prediction,
        raw_prediction,
    )


# =============================================================================
# Hydrology-informed dynamic prior
# =============================================================================

def groundwater_forcing_index(
    predicted_groundwater: np.ndarray,
    activation_depth: float,
    normalization_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Calculate the activation head and normalized groundwater forcing."""

    predicted_groundwater = require_finite_array(
        predicted_groundwater,
        "Predicted groundwater depth",
    )

    if not np.isfinite(activation_depth):
        raise ValueError(
            "The groundwater activation depth must be finite."
        )

    effective_head = np.maximum(
        activation_depth
        - predicted_groundwater,
        0.0,
    )

    squared_head = (
        effective_head ** 2
    )

    if normalization_scale is None:
        positive_squared_head = squared_head[
            squared_head > 0.0
        ]

        if len(positive_squared_head) > 0:
            normalization_scale = float(
                np.quantile(
                    positive_squared_head,
                    0.95,
                )
            )
        else:
            normalization_scale = 1.0

    normalization_scale = max(
        float(normalization_scale),
        1.0e-12,
    )

    normalized_forcing = (
        squared_head
        / normalization_scale
    )

    return (
        require_finite_array(
            effective_head,
            "Effective groundwater activation head",
            expected_length=len(predicted_groundwater),
        ),
        require_finite_array(
            normalized_forcing,
            "Normalized groundwater forcing",
            expected_length=len(predicted_groundwater),
        ),
        normalization_scale,
    )


def fit_hydrology_informed_prior(
    development_data: pd.DataFrame,
    predicted_groundwater: np.ndarray,
    activation_depth: float,
) -> dict[str, Any]:
    """
    Fit the bounded robust hydrology-informed dynamic velocity prior.

    The prior is

        rho * current latent velocity
        + mean daily drift
        + groundwater coefficient * normalized groundwater forcing.
    """

    predicted_groundwater = require_finite_array(
        predicted_groundwater,
        "Prior-training groundwater predictions",
        expected_length=len(development_data),
    )

    (
        _,
        normalized_forcing,
        normalization_scale,
    ) = groundwater_forcing_index(
        predicted_groundwater=predicted_groundwater,
        activation_depth=activation_depth,
        normalization_scale=None,
    )

    reference_target = require_finite_array(
        development_data[
            "target_velocity"
        ],
        "Prior-training reference velocity",
        expected_length=len(development_data),
    )

    current_velocity = require_finite_array(
        development_data[
            "latent_velocity"
        ],
        "Prior-training current velocity",
        expected_length=len(development_data),
    )

    target_scale = robust_scale(
        reference_target,
        minimum=0.03,
    )

    def objective(
        parameters: np.ndarray,
    ) -> np.ndarray:
        rho, intercept, groundwater_coefficient = (
            parameters
        )

        prediction = (
            rho * current_velocity
            + intercept
            + groundwater_coefficient
            * normalized_forcing
        )

        return (
            prediction - reference_target
        ) / target_scale

    optimization = least_squares(
        objective,
        x0=PRIOR_INITIAL_PARAMETERS,
        bounds=(
            PRIOR_LOWER_BOUNDS,
            PRIOR_UPPER_BOUNDS,
        ),
        loss="soft_l1",
        f_scale=0.50,
        max_nfev=3000,
    )

    if not optimization.success:
        raise RuntimeError(
            "Hydrology-informed prior optimization failed: "
            f"{optimization.message}"
        )

    optimized_parameters = require_finite_array(
        optimization.x,
        "Optimized hydrology-prior parameters",
        expected_length=3,
    )

    return {
        "parameters": optimized_parameters,
        "activation_depth": float(
            activation_depth
        ),
        "normalization_scale": float(
            normalization_scale
        ),
        "success": bool(
            optimization.success
        ),
        "status": int(
            optimization.status
        ),
        "cost": float(
            optimization.cost
        ),
        "optimality": float(
            optimization.optimality
        ),
        "message": str(
            optimization.message
        ),
    }


def predict_hydrology_informed_prior(
    bundle: dict[str, Any],
    data: pd.DataFrame,
    predicted_groundwater: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Generate hydrology-informed dynamic-prior predictions."""

    parameters = require_finite_array(
        bundle["parameters"],
        "Hydrology-prior parameters",
        expected_length=3,
    )

    rho, intercept, groundwater_coefficient = (
        parameters
    )

    predicted_groundwater = require_finite_array(
        predicted_groundwater,
        "Prior-prediction groundwater depth",
        expected_length=len(data),
    )

    (
        effective_head,
        normalized_forcing,
        _,
    ) = groundwater_forcing_index(
        predicted_groundwater=predicted_groundwater,
        activation_depth=bundle[
            "activation_depth"
        ],
        normalization_scale=bundle[
            "normalization_scale"
        ],
    )

    current_velocity = require_finite_array(
        data["latent_velocity"],
        "Current latent velocity",
        expected_length=len(data),
    )

    prior_prediction = (
        rho * current_velocity
        + intercept
        + groundwater_coefficient
        * normalized_forcing
    )

    return (
        require_finite_array(
            prior_prediction,
            "Hydrology-informed prior predictions",
            expected_length=len(data),
        ),
        effective_head,
        normalized_forcing,
    )


# =============================================================================
# Residual-learning predictor matrix
# =============================================================================

def construct_residual_matrix(
    data: pd.DataFrame,
    predicted_groundwater: np.ndarray,
    hydrology_prior: np.ndarray,
    effective_head: np.ndarray,
    normalized_forcing: np.ndarray,
) -> pd.DataFrame:
    """Construct the complete RSRL-HU residual-learning predictor matrix."""

    number_of_rows = len(data)

    predicted_groundwater = require_finite_array(
        predicted_groundwater,
        "Residual-matrix groundwater predictions",
        expected_length=number_of_rows,
    )

    hydrology_prior = require_finite_array(
        hydrology_prior,
        "Residual-matrix dynamic-prior predictions",
        expected_length=number_of_rows,
    )

    effective_head = require_finite_array(
        effective_head,
        "Residual-matrix activation heads",
        expected_length=number_of_rows,
    )

    normalized_forcing = require_finite_array(
        normalized_forcing,
        "Residual-matrix groundwater forcing",
        expected_length=number_of_rows,
    )

    matrix = pd.DataFrame(
        index=np.arange(
            number_of_rows,
            dtype=int,
        )
    )

    matrix["Hydrology_prior"] = (
        hydrology_prior
    )

    for feature in CORE_STATE_FEATURES:
        matrix[feature] = data[
            feature
        ].to_numpy(dtype=float)

    for feature in STATE_UNCERTAINTY_FEATURES:
        matrix[feature] = data[
            feature
        ].to_numpy(dtype=float)

    for feature in GROUNDWATER_HISTORY_FEATURES:
        matrix[feature] = data[
            feature
        ].to_numpy(dtype=float)

    matrix["Predicted_GL_next"] = (
        predicted_groundwater
    )

    matrix["Predicted_GWR_next"] = (
        data["GL"].to_numpy(dtype=float)
        - predicted_groundwater
    )

    matrix["Effective_head"] = (
        effective_head
    )

    matrix["Pressure_scaled"] = (
        normalized_forcing
    )

    matrix["GL_anomaly7"] = (
        predicted_groundwater
        - data[
            "GL_mean7"
        ].to_numpy(dtype=float)
    )

    if list(matrix.columns) != RESIDUAL_FEATURES:
        raise RuntimeError(
            "The residual-predictor order is inconsistent with the "
            "predefined RSRL-HU feature structure."
        )

    if matrix.shape != (
        number_of_rows,
        len(RESIDUAL_FEATURES),
    ):
        raise RuntimeError(
            "The residual-predictor matrix has an unexpected shape."
        )

    if not np.all(
        np.isfinite(
            matrix.to_numpy(dtype=float)
        )
    ):
        raise RuntimeError(
            "The residual-predictor matrix contains non-finite values."
        )

    return matrix


# =============================================================================
# Nested temporal model selection
# =============================================================================

def select_residual_configuration(
    raw: pd.DataFrame,
    development_rows: np.ndarray,
    filter_config: dict[str, float],
    activation_depth: float,
) -> dict[str, Any]:
    """
    Select residual-model complexity and shrinkage through nested temporal
    validation.
    """

    development_rows = np.asarray(
        development_rows,
        dtype=int,
    ).reshape(-1)

    if len(development_rows) == 0:
        raise ValueError(
            "The development-row sequence is empty."
        )

    if not np.all(
        np.diff(development_rows) > 0
    ):
        raise ValueError(
            "Development rows must be strictly increasing."
        )

    minimum_required = (
        INNER_SPLITS * INNER_TEST_SIZE
        + INNER_GAP
        + 1
    )

    if len(development_rows) < minimum_required:
        raise RuntimeError(
            "The development period is too short for the prespecified "
            "nested temporal validation."
        )

    splitter = TimeSeriesSplit(
        n_splits=INNER_SPLITS,
        test_size=INNER_TEST_SIZE,
        gap=INNER_GAP,
    )

    fold_cache = []

    for inner_fold, (
        training_indices,
        validation_indices,
    ) in enumerate(
        splitter.split(development_rows),
        start=1,
    ):
        training_rows = development_rows[
            training_indices
        ]

        validation_rows = development_rows[
            validation_indices
        ]

        if training_rows[-1] >= validation_rows[0]:
            raise RuntimeError(
                "An inner training period overlaps its validation period."
            )

        samples, _, _ = prepare_dataset(
            raw=raw,
            filter_training_end=int(
                training_rows[-1]
            ),
            run_end=int(
                validation_rows[-1]
            ),
            filter_config=filter_config,
        )

        training = select_target_rows(
            samples,
            training_rows,
        )

        validation = select_target_rows(
            samples,
            validation_rows,
        )

        (
            groundwater_bundle,
            development_groundwater_oof,
            valid_groundwater,
        ) = fit_groundwater_forecaster(
            training
        )

        valid_positions = np.flatnonzero(
            valid_groundwater
        )

        residual_training = (
            training.iloc[
                valid_positions
            ]
            .copy()
            .reset_index(drop=True)
        )

        residual_training_groundwater = (
            development_groundwater_oof[
                valid_groundwater
            ]
        )

        residual_training_groundwater = (
            require_finite_array(
                residual_training_groundwater,
                "Inner temporally external groundwater predictions",
                expected_length=len(
                    residual_training
                ),
            )
        )

        (
            validation_groundwater,
            _,
        ) = predict_groundwater(
            groundwater_bundle,
            validation,
        )

        prior_bundle = fit_hydrology_informed_prior(
            development_data=(
                residual_training
            ),
            predicted_groundwater=(
                residual_training_groundwater
            ),
            activation_depth=(
                activation_depth
            ),
        )

        (
            training_prior,
            training_head,
            training_forcing,
        ) = predict_hydrology_informed_prior(
            bundle=prior_bundle,
            data=residual_training,
            predicted_groundwater=(
                residual_training_groundwater
            ),
        )

        (
            validation_prior,
            validation_head,
            validation_forcing,
        ) = predict_hydrology_informed_prior(
            bundle=prior_bundle,
            data=validation,
            predicted_groundwater=(
                validation_groundwater
            ),
        )

        training_matrix = construct_residual_matrix(
            data=residual_training,
            predicted_groundwater=(
                residual_training_groundwater
            ),
            hydrology_prior=training_prior,
            effective_head=training_head,
            normalized_forcing=training_forcing,
        )

        validation_matrix = construct_residual_matrix(
            data=validation,
            predicted_groundwater=(
                validation_groundwater
            ),
            hydrology_prior=validation_prior,
            effective_head=validation_head,
            normalized_forcing=validation_forcing,
        )

        residual_target = (
            residual_training[
                "target_velocity"
            ].to_numpy(dtype=float)
            - training_prior
        )

        residual_target = require_finite_array(
            residual_target,
            "Inner residual-learning target",
            expected_length=len(
                residual_training
            ),
        )

        validation_target = require_finite_array(
            validation[
                "target_velocity"
            ],
            "Inner validation reference velocity",
            expected_length=len(validation),
        )

        fold_cache.append(
            {
                "inner_fold": inner_fold,
                "training_matrix": training_matrix,
                "validation_matrix": validation_matrix,
                "residual_target": residual_target,
                "validation_target": validation_target,
                "validation_prior": validation_prior,
            }
        )

    if len(fold_cache) != INNER_SPLITS:
        raise RuntimeError(
            "Nested temporal validation produced an unexpected number "
            "of folds."
        )

    search_records = []

    for configuration_id, parameters in enumerate(
        RESIDUAL_CONFIGURATIONS
    ):
        fold_residual_predictions = []

        for fold_data in fold_cache:
            residual_model = make_huber_gradient_boosting(
                parameters
            )

            residual_model.fit(
                fold_data["training_matrix"],
                fold_data["residual_target"],
            )

            predicted_residual = residual_model.predict(
                fold_data[
                    "validation_matrix"
                ]
            )

            predicted_residual = require_finite_array(
                predicted_residual,
                "Inner validation residual predictions",
                expected_length=len(
                    fold_data[
                        "validation_target"
                    ]
                ),
            )

            fold_residual_predictions.append(
                predicted_residual
            )

        for shrinkage in RESIDUAL_SHRINKAGE_GRID:
            shrinkage = float(
                np.clip(
                    shrinkage,
                    0.0,
                    1.0,
                )
            )

            fold_scores = []

            for fold_data, predicted_residual in zip(
                fold_cache,
                fold_residual_predictions,
            ):
                prediction = (
                    fold_data[
                        "validation_prior"
                    ]
                    + shrinkage
                    * predicted_residual
                )

                fold_scores.append(
                    temporal_selection_score(
                        fold_data[
                            "validation_target"
                        ],
                        prediction,
                    )
                )

            search_records.append(
                {
                    "configuration_id": configuration_id,
                    "configuration": parameters[
                        "configuration"
                    ],
                    "shrinkage": shrinkage,
                    "mean_score": float(
                        np.mean(fold_scores)
                    ),
                    "score_fold1": float(
                        fold_scores[0]
                    ),
                    "score_fold2": float(
                        fold_scores[1]
                    ),
                    "score_fold3": float(
                        fold_scores[2]
                    ),
                }
            )

    search_table = (
        pd.DataFrame(search_records)
        .sort_values(
            [
                "mean_score",
                "shrinkage",
                "configuration_id",
            ],
            ascending=[
                True,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    if len(search_table) == 0:
        raise RuntimeError(
            "Residual-model selection produced no candidate result."
        )

    selected_row = search_table.iloc[0]

    configuration_id = int(
        selected_row[
            "configuration_id"
        ]
    )

    selected_parameters = deepcopy(
        RESIDUAL_CONFIGURATIONS[
            configuration_id
        ]
    )

    return {
        "configuration_id": configuration_id,
        "configuration": selected_parameters[
            "configuration"
        ],
        "parameters": selected_parameters,
        "shrinkage": float(
            np.clip(
                selected_row["shrinkage"],
                0.0,
                1.0,
            )
        ),
        "score": float(
            selected_row["mean_score"]
        ),
        "search": search_table,
    }


# =============================================================================
# Complete RSRL-HU model fitting and prediction
# =============================================================================

def fit_rsrl_hu(
    development_data: pd.DataFrame,
    selection: dict[str, Any],
    activation_depth: float,
) -> dict[str, Any]:
    """Fit the complete RSRL-HU point-forecast model."""

    if len(development_data) <= HYDRO_WARMUP:
        raise RuntimeError(
            "The development period is too short to fit RSRL-HU."
        )

    (
        groundwater_bundle,
        development_groundwater_oof,
        valid_groundwater,
    ) = fit_groundwater_forecaster(
        development_data
    )

    valid_positions = np.flatnonzero(
        valid_groundwater
    )

    residual_training = (
        development_data.iloc[
            valid_positions
        ]
        .copy()
        .reset_index(drop=True)
    )

    residual_training_groundwater = (
        development_groundwater_oof[
            valid_groundwater
        ]
    )

    residual_training_groundwater = require_finite_array(
        residual_training_groundwater,
        "Final temporally external groundwater predictions",
        expected_length=len(
            residual_training
        ),
    )

    prior_bundle = fit_hydrology_informed_prior(
        development_data=(
            residual_training
        ),
        predicted_groundwater=(
            residual_training_groundwater
        ),
        activation_depth=activation_depth,
    )

    (
        training_prior,
        training_head,
        training_forcing,
    ) = predict_hydrology_informed_prior(
        bundle=prior_bundle,
        data=residual_training,
        predicted_groundwater=(
            residual_training_groundwater
        ),
    )

    residual_matrix = construct_residual_matrix(
        data=residual_training,
        predicted_groundwater=(
            residual_training_groundwater
        ),
        hydrology_prior=training_prior,
        effective_head=training_head,
        normalized_forcing=training_forcing,
    )

    residual_target = (
        residual_training[
            "target_velocity"
        ].to_numpy(dtype=float)
        - training_prior
    )

    residual_target = require_finite_array(
        residual_target,
        "Final residual-learning target",
        expected_length=len(
            residual_training
        ),
    )

    residual_model = make_huber_gradient_boosting(
        selection["parameters"]
    )

    residual_model.fit(
        residual_matrix,
        residual_target,
    )

    return {
        "groundwater": groundwater_bundle,
        "prior": prior_bundle,
        "residual_model": residual_model,
        "residual_shrinkage": float(
            np.clip(
                selection["shrinkage"],
                0.0,
                1.0,
            )
        ),
        "selection": deepcopy(selection),
        "development_size": int(
            len(development_data)
        ),
        "residual_training_size": int(
            len(residual_training)
        ),
    }


def predict_rsrl_hu(
    bundle: dict[str, Any],
    data: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Generate complete RSRL-HU point forecasts."""

    if len(data) == 0:
        raise ValueError(
            "No samples were supplied for RSRL-HU prediction."
        )

    (
        predicted_groundwater,
        raw_predicted_groundwater,
    ) = predict_groundwater(
        bundle["groundwater"],
        data,
    )

    (
        prior_prediction,
        effective_head,
        normalized_forcing,
    ) = predict_hydrology_informed_prior(
        bundle=bundle["prior"],
        data=data,
        predicted_groundwater=(
            predicted_groundwater
        ),
    )

    residual_matrix = construct_residual_matrix(
        data=data,
        predicted_groundwater=(
            predicted_groundwater
        ),
        hydrology_prior=prior_prediction,
        effective_head=effective_head,
        normalized_forcing=(
            normalized_forcing
        ),
    )

    predicted_residual = (
        bundle[
            "residual_model"
        ].predict(
            residual_matrix
        )
    )

    predicted_residual = require_finite_array(
        predicted_residual,
        "Predicted residual corrections",
        expected_length=len(data),
    )

    residual_shrinkage = float(
        np.clip(
            bundle[
                "residual_shrinkage"
            ],
            0.0,
            1.0,
        )
    )

    final_prediction = (
        prior_prediction
        + residual_shrinkage
        * predicted_residual
    )

    outputs = {
        "prediction": final_prediction,
        "hydrology_prior": prior_prediction,
        "predicted_residual": predicted_residual,
        "predicted_groundwater": predicted_groundwater,
        "raw_predicted_groundwater": raw_predicted_groundwater,
        "effective_head": effective_head,
        "normalized_forcing": normalized_forcing,
    }

    for name, values in outputs.items():
        outputs[name] = require_finite_array(
            values,
            name,
            expected_length=len(data),
        )

    return outputs


# =============================================================================
# Rolling prequential uncertainty quantification
# =============================================================================

def rolling_prequential_intervals(
    calibration_reference: np.ndarray,
    calibration_prediction: np.ndarray,
    test_reference: np.ndarray,
    test_prediction: np.ndarray,
    alpha: float,
    window: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Construct symmetric rolling prequential empirical prediction intervals.

    Before each external target is observed, its interval uses only initial
    calibration errors and errors from preceding external forecasts.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError(
            "The interval miscoverage rate must lie strictly between zero "
            "and one."
        )

    if int(window) <= 0:
        raise ValueError(
            "The prequential error-memory window must be positive."
        )

    calibration_reference = require_finite_array(
        calibration_reference,
        "Calibration reference velocities",
    )

    calibration_prediction = require_finite_array(
        calibration_prediction,
        "Calibration velocity predictions",
        expected_length=len(
            calibration_reference
        ),
    )

    test_reference = require_finite_array(
        test_reference,
        "External reference velocities",
    )

    test_prediction = require_finite_array(
        test_prediction,
        "External velocity predictions",
        expected_length=len(
            test_reference
        ),
    )

    if len(calibration_reference) == 0:
        raise RuntimeError(
            "At least one calibration error is required for interval "
            "construction."
        )

    error_history = list(
        np.abs(
            calibration_reference
            - calibration_prediction
        )
    )

    lower = np.zeros(
        len(test_reference),
        dtype=float,
    )

    upper = np.zeros(
        len(test_reference),
        dtype=float,
    )

    quantiles = np.zeros(
        len(test_reference),
        dtype=float,
    )

    for position in range(
        len(test_reference)
    ):
        available_errors = error_history[
            -int(window):
        ]

        quantile = finite_sample_error_quantile(
            available_errors,
            alpha=alpha,
        )

        quantiles[position] = quantile

        lower[position] = (
            test_prediction[position]
            - quantile
        )

        upper[position] = (
            test_prediction[position]
            + quantile
        )

        # The current error becomes available only after the current
        # prediction interval has been issued.
        error_history.append(
            abs(
                test_reference[position]
                - test_prediction[position]
            )
        )

    return (
        lower,
        upper,
        quantiles,
    )


# =============================================================================
# Prespecified temporal partitions
# =============================================================================

def rows_between_dates(
    raw: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> np.ndarray:
    """Return raw target rows between two inclusive target dates."""

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if start > end:
        raise ValueError(
            "The period start date follows the period end date."
        )

    mask = (
        (raw["Date"] >= start)
        & (raw["Date"] <= end)
    )

    rows = raw.index[
        mask
    ].to_numpy(dtype=int)

    rows = rows[
        rows >= SAMPLE_START_ROW
    ]

    if len(rows) == 0:
        raise RuntimeError(
            "No eligible target rows were found between "
            f"{start.date()} and {end.date()}."
        )

    return rows


def construct_temporal_design(
    raw: pd.DataFrame,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Construct the manuscript-prespecified external evaluation design."""

    rolling_specifications = [
        {
            "evaluation": "Rolling-origin evaluation",
            "fold": 1,
            "development_rows": rows_between_dates(
                raw,
                "2016-02-02",
                "2017-02-18",
            ),
            "calibration_rows": rows_between_dates(
                raw,
                "2017-02-19",
                "2017-04-04",
            ),
            "test_rows": rows_between_dates(
                raw,
                "2017-04-06",
                "2017-05-20",
            ),
            "gap_days": 1,
        },
        {
            "evaluation": "Rolling-origin evaluation",
            "fold": 2,
            "development_rows": rows_between_dates(
                raw,
                "2016-02-02",
                "2017-04-04",
            ),
            "calibration_rows": rows_between_dates(
                raw,
                "2017-04-05",
                "2017-05-19",
            ),
            "test_rows": rows_between_dates(
                raw,
                "2017-05-21",
                "2017-07-04",
            ),
            "gap_days": 1,
        },
        {
            "evaluation": "Rolling-origin evaluation",
            "fold": 3,
            "development_rows": rows_between_dates(
                raw,
                "2016-02-02",
                "2017-05-19",
            ),
            "calibration_rows": rows_between_dates(
                raw,
                "2017-05-20",
                "2017-07-03",
            ),
            "test_rows": rows_between_dates(
                raw,
                "2017-07-05",
                "2017-08-18",
            ),
            "gap_days": 1,
        },
        {
            "evaluation": "Rolling-origin evaluation",
            "fold": 4,
            "development_rows": rows_between_dates(
                raw,
                "2016-02-02",
                "2017-07-03",
            ),
            "calibration_rows": rows_between_dates(
                raw,
                "2017-07-04",
                "2017-08-17",
            ),
            "test_rows": rows_between_dates(
                raw,
                "2017-08-19",
                "2017-10-02",
            ),
            "gap_days": 1,
        },
    ]

    late_specification = {
        "evaluation": "Late-period validation",
        "fold": 1,
        "development_rows": rows_between_dates(
            raw,
            "2016-02-02",
            "2017-08-18",
        ),
        "calibration_rows": rows_between_dates(
            raw,
            "2017-08-19",
            "2017-10-02",
        ),
        "test_rows": rows_between_dates(
            raw,
            "2017-10-03",
            "2017-12-31",
        ),
        "gap_days": 0,
    }

    expected_rolling_development_sizes = [
        383,
        428,
        473,
        518,
    ]

    for index, specification in enumerate(
        rolling_specifications
    ):
        development_rows = specification[
            "development_rows"
        ]

        calibration_rows = specification[
            "calibration_rows"
        ]

        test_rows = specification[
            "test_rows"
        ]

        if len(development_rows) != (
            expected_rolling_development_sizes[
                index
            ]
        ):
            raise RuntimeError(
                "A rolling development-period size is inconsistent with "
                "the prespecified design."
            )

        if len(calibration_rows) != 45:
            raise RuntimeError(
                "A rolling calibration period does not contain 45 targets."
            )

        if len(test_rows) != 45:
            raise RuntimeError(
                "A rolling external period does not contain 45 targets."
            )

        if development_rows[-1] >= calibration_rows[0]:
            raise RuntimeError(
                "A rolling development period overlaps its calibration "
                "period."
            )

        if calibration_rows[-1] >= test_rows[0]:
            raise RuntimeError(
                "A rolling calibration period overlaps its external period."
            )

        actual_gap = int(
            test_rows[0]
            - calibration_rows[-1]
            - 1
        )

        if actual_gap != specification[
            "gap_days"
        ]:
            raise RuntimeError(
                "A rolling block has an incorrect calibration-to-test gap."
            )

    if len(
        late_specification[
            "development_rows"
        ]
    ) != 564:
        raise RuntimeError(
            "The late-period development set does not contain 564 targets."
        )

    if len(
        late_specification[
            "calibration_rows"
        ]
    ) != 45:
        raise RuntimeError(
            "The late-period calibration set does not contain 45 targets."
        )

    if len(
        late_specification[
            "test_rows"
        ]
    ) != 90:
        raise RuntimeError(
            "The late-period validation set does not contain 90 targets."
        )

    return (
        rolling_specifications,
        late_specification,
    )


# =============================================================================
# External evaluation
# =============================================================================

def evaluate_external_block(
    raw: pd.DataFrame,
    specification: dict[str, Any],
    filter_config: dict[str, float],
    activation_depth: float,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    """Fit and evaluate one frozen external RSRL-HU model."""

    development_rows = np.asarray(
        specification[
            "development_rows"
        ],
        dtype=int,
    )

    calibration_rows = np.asarray(
        specification[
            "calibration_rows"
        ],
        dtype=int,
    )

    test_rows = np.asarray(
        specification[
            "test_rows"
        ],
        dtype=int,
    )

    if development_rows[-1] >= calibration_rows[0]:
        raise RuntimeError(
            "Development and calibration rows overlap."
        )

    if calibration_rows[-1] >= test_rows[0]:
        raise RuntimeError(
            "Calibration and external rows overlap."
        )

    selection = select_residual_configuration(
        raw=raw,
        development_rows=development_rows,
        filter_config=filter_config,
        activation_depth=activation_depth,
    )

    (
        samples,
        _,
        filter_parameters,
    ) = prepare_dataset(
        raw=raw,
        filter_training_end=int(
            development_rows[-1]
        ),
        run_end=int(
            test_rows[-1]
        ),
        filter_config=filter_config,
    )

    development = select_target_rows(
        samples,
        development_rows,
    )

    calibration = select_target_rows(
        samples,
        calibration_rows,
    )

    test = select_target_rows(
        samples,
        test_rows,
    )

    fitted_model = fit_rsrl_hu(
        development_data=development,
        selection=selection,
        activation_depth=activation_depth,
    )

    calibration_output = predict_rsrl_hu(
        fitted_model,
        calibration,
    )

    test_output = predict_rsrl_hu(
        fitted_model,
        test,
    )

    calibration_reference = require_finite_array(
        calibration[
            "target_velocity"
        ],
        "Calibration reference velocities",
        expected_length=len(calibration),
    )

    test_reference = require_finite_array(
        test[
            "target_velocity"
        ],
        "External reference velocities",
        expected_length=len(test),
    )

    (
        lower,
        upper,
        quantiles,
    ) = rolling_prequential_intervals(
        calibration_reference=(
            calibration_reference
        ),
        calibration_prediction=(
            calibration_output[
                "prediction"
            ]
        ),
        test_reference=test_reference,
        test_prediction=test_output[
            "prediction"
        ],
        alpha=INTERVAL_ALPHA,
        window=ERROR_MEMORY_WINDOW,
    )

    predictions = pd.DataFrame(
        {
            "Evaluation": specification[
                "evaluation"
            ],
            "Fold": specification["fold"],
            "Origin_date": test[
                "Date"
            ].to_numpy(),
            "Target_date": test[
                "target_date"
            ].to_numpy(),
            "Target_row": test[
                "target_row"
            ].to_numpy(dtype=int),
            "Reference_velocity": test_reference,
            "Predicted_velocity": test_output[
                "prediction"
            ],
            "Hydrology_prior": test_output[
                "hydrology_prior"
            ],
            "Predicted_residual": test_output[
                "predicted_residual"
            ],
            "Lower_90": lower,
            "Upper_90": upper,
            "Prequential_qhat": quantiles,
            "Current_GL": test[
                "GL"
            ].to_numpy(dtype=float),
            "Reference_GL_next": test[
                "target_GL"
            ].to_numpy(dtype=float),
            "Reference_GL_delta": test[
                "target_GL_delta"
            ].to_numpy(dtype=float),
            "Predicted_GL_raw": test_output[
                "raw_predicted_groundwater"
            ],
            "Predicted_GL_next": test_output[
                "predicted_groundwater"
            ],
            "Effective_head": test_output[
                "effective_head"
            ],
            "Pressure_scaled": test_output[
                "normalized_forcing"
            ],
        }
    )

    rho, intercept, coefficient = (
        fitted_model[
            "prior"
        ]["parameters"]
    )

    summary = {
        "evaluation": specification[
            "evaluation"
        ],
        "fold": int(
            specification["fold"]
        ),
        "development_start": raw.loc[
            development_rows[0],
            "Date",
        ],
        "development_end": raw.loc[
            development_rows[-1],
            "Date",
        ],
        "development_size": int(
            len(development_rows)
        ),
        "calibration_start": raw.loc[
            calibration_rows[0],
            "Date",
        ],
        "calibration_end": raw.loc[
            calibration_rows[-1],
            "Date",
        ],
        "calibration_size": int(
            len(calibration_rows)
        ),
        "test_start": raw.loc[
            test_rows[0],
            "Date",
        ],
        "test_end": raw.loc[
            test_rows[-1],
            "Date",
        ],
        "test_size": int(
            len(test_rows)
        ),
        "gap_days": int(
            specification[
                "gap_days"
            ]
        ),
        "residual_configuration": (
            selection[
                "configuration"
            ]
        ),
        "residual_shrinkage": float(
            selection[
                "shrinkage"
            ]
        ),
        "residual_selection_score": float(
            selection[
                "score"
            ]
        ),
        "groundwater_shrinkage": float(
            fitted_model[
                "groundwater"
            ]["shrinkage"]
        ),
        "prior_rho": float(rho),
        "prior_intercept": float(
            intercept
        ),
        "groundwater_coefficient": float(
            coefficient
        ),
        "groundwater_activation_fraction": float(
            np.mean(
                test_output[
                    "effective_head"
                ] > 0.0
            )
        ),
        "observation_scale": float(
            filter_parameters[
                "observation_scale"
            ]
        ),
    }

    return (
        predictions,
        summary,
    )


# =============================================================================
# Evaluation summaries
# =============================================================================

def summarize_velocity_forecasts(
    predictions: pd.DataFrame,
    evaluation_name: str,
) -> pd.DataFrame:
    """Summarize RSRL-HU point and interval performance."""

    reference = predictions[
        "Reference_velocity"
    ].to_numpy(dtype=float)

    prediction = predictions[
        "Predicted_velocity"
    ].to_numpy(dtype=float)

    lower = predictions[
        "Lower_90"
    ].to_numpy(dtype=float)

    upper = predictions[
        "Upper_90"
    ].to_numpy(dtype=float)

    point = point_metrics(
        reference,
        prediction,
    )

    interval = prediction_interval_metrics(
        reference,
        lower,
        upper,
        alpha=INTERVAL_ALPHA,
    )

    return pd.DataFrame(
        [
            {
                "Evaluation": evaluation_name,
                "N": point["N"],
                "RMSE_mm_d": point["RMSE"],
                "MAE_mm_d": point["MAE"],
                "R2": point["R2"],
                "Bias_mm_d": point["Bias"],
                "Covered_targets": interval[
                    "Covered"
                ],
                "PICP_percent": (
                    100.0
                    * interval["PICP"]
                ),
                "MPIW_mm_d": interval[
                    "MPIW"
                ],
                "Median_width_mm_d": interval[
                    "Median_width"
                ],
                "Winkler_mm_d": interval[
                    "Winkler"
                ],
            }
        ]
    )


def summarize_groundwater_forecasts(
    predictions: pd.DataFrame,
    evaluation_name: str,
) -> pd.DataFrame:
    """Summarize following-day groundwater-depth-change forecasting."""

    reference_change = predictions[
        "Reference_GL_delta"
    ].to_numpy(dtype=float)

    current_groundwater = predictions[
        "Current_GL"
    ].to_numpy(dtype=float)

    persistence_change = np.zeros(
        len(predictions),
        dtype=float,
    )

    raw_change = (
        predictions[
            "Predicted_GL_raw"
        ].to_numpy(dtype=float)
        - current_groundwater
    )

    shrunk_change = (
        predictions[
            "Predicted_GL_next"
        ].to_numpy(dtype=float)
        - current_groundwater
    )

    persistence_metrics = point_metrics(
        reference_change,
        persistence_change,
    )

    raw_metrics = point_metrics(
        reference_change,
        raw_change,
    )

    shrunk_metrics = point_metrics(
        reference_change,
        shrunk_change,
    )

    rows = []

    for model_label, metrics in [
        (
            "Groundwater persistence",
            persistence_metrics,
        ),
        (
            "Raw Huber-GBR",
            raw_metrics,
        ),
        (
            "Persistence-shrunk Huber-GBR",
            shrunk_metrics,
        ),
    ]:
        rows.append(
            {
                "Evaluation": evaluation_name,
                "Groundwater_model": model_label,
                "N": metrics["N"],
                "RMSE_m": metrics["RMSE"],
                "MAE_m": metrics["MAE"],
                "R2": metrics["R2"],
                "Bias_m": metrics["Bias"],
                "RMSE_skill_vs_persistence_percent": (
                    percentage_skill(
                        candidate_error=metrics[
                            "RMSE"
                        ],
                        reference_error=(
                            persistence_metrics[
                                "RMSE"
                            ]
                        ),
                    )
                ),
                "MAE_skill_vs_persistence_percent": (
                    percentage_skill(
                        candidate_error=metrics[
                            "MAE"
                        ],
                        reference_error=(
                            persistence_metrics[
                                "MAE"
                            ]
                        ),
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# Operational following-day forecast
# =============================================================================

def generate_operational_forecast(
    raw: pd.DataFrame,
    locked_selection: dict[str, Any],
    filter_config: dict[str, float],
    activation_depth: float,
) -> pd.DataFrame:
    """
    Generate a following-day forecast using one frozen point-forecast model.

    The final 45 available reference targets are reserved for interval
    calibration. The point-forecast model is fitted on the preceding
    development period. The calibration predictions and operational forecast
    center are therefore generated by the same frozen fitted mapping.
    """

    all_target_rows = np.arange(
        SAMPLE_START_ROW,
        len(raw),
        dtype=int,
    )

    calibration_size = 45

    if len(all_target_rows) <= calibration_size:
        raise RuntimeError(
            "Insufficient observations are available for operational "
            "calibration."
        )

    development_rows = all_target_rows[
        :-calibration_size
    ]

    calibration_rows = all_target_rows[
        -calibration_size:
    ]

    if development_rows[-1] >= calibration_rows[0]:
        raise RuntimeError(
            "Operational development and calibration periods overlap."
        )

    (
        samples,
        features,
        _,
    ) = prepare_dataset(
        raw=raw,
        filter_training_end=int(
            development_rows[-1]
        ),
        run_end=len(raw) - 1,
        filter_config=filter_config,
    )

    development = select_target_rows(
        samples,
        development_rows,
    )

    calibration = select_target_rows(
        samples,
        calibration_rows,
    )

    frozen_model = fit_rsrl_hu(
        development_data=development,
        selection=locked_selection,
        activation_depth=activation_depth,
    )

    calibration_output = predict_rsrl_hu(
        frozen_model,
        calibration,
    )

    calibration_errors = np.abs(
        calibration[
            "target_velocity"
        ].to_numpy(dtype=float)
        - calibration_output[
            "prediction"
        ]
    )

    calibration_errors = require_finite_array(
        calibration_errors,
        "Operational calibration errors",
        expected_length=len(calibration),
    )

    qhat = finite_sample_error_quantile(
        calibration_errors[
            -ERROR_MEMORY_WINDOW:
        ],
        alpha=INTERVAL_ALPHA,
    )

    latest_origin = (
        features.iloc[[-1]]
        .copy()
        .reset_index(drop=True)
    )

    required_latest_features = (
        CORE_STATE_FEATURES
        + STATE_UNCERTAINTY_FEATURES
        + HYDROLOGICAL_FEATURES
    )

    if latest_origin[
        required_latest_features
    ].isna().any().any():
        raise RuntimeError(
            "The final forecast origin does not contain every predictor "
            "required for operational forecasting."
        )

    latest_output = predict_rsrl_hu(
        frozen_model,
        latest_origin,
    )

    point_forecast = float(
        latest_output[
            "prediction"
        ][0]
    )

    lower = (
        point_forecast - qhat
    )

    upper = (
        point_forecast + qhat
    )

    latest_latent_displacement = float(
        latest_origin[
            "latent_SD"
        ].iloc[0]
    )

    return pd.DataFrame(
        [
            {
                "Model": MODEL_NAME,
                "Model_version": MODEL_VERSION,
                "Forecast_origin": latest_origin[
                    "Date"
                ].iloc[0],
                "Forecast_target": (
                    latest_origin[
                        "Date"
                    ].iloc[0]
                    + pd.Timedelta(days=1)
                ),
                "Current_latent_velocity_mm_d": float(
                    latest_origin[
                        "latent_velocity"
                    ].iloc[0]
                ),
                "Predicted_groundwater_raw_m": float(
                    latest_output[
                        "raw_predicted_groundwater"
                    ][0]
                ),
                "Predicted_groundwater_m": float(
                    latest_output[
                        "predicted_groundwater"
                    ][0]
                ),
                "Groundwater_shrinkage": float(
                    frozen_model[
                        "groundwater"
                    ]["shrinkage"]
                ),
                "Hydrology_prior_mm_d": float(
                    latest_output[
                        "hydrology_prior"
                    ][0]
                ),
                "Residual_configuration": (
                    frozen_model[
                        "selection"
                    ]["configuration"]
                ),
                "Residual_shrinkage": float(
                    frozen_model[
                        "residual_shrinkage"
                    ]
                ),
                "Predicted_velocity_mm_d": point_forecast,
                "Lower_90_mm_d": lower,
                "Upper_90_mm_d": upper,
                "Predicted_latent_SD_mm": (
                    latest_latent_displacement
                    + point_forecast
                ),
                "Latent_SD_lower_90_mm": (
                    latest_latent_displacement
                    + lower
                ),
                "Latent_SD_upper_90_mm": (
                    latest_latent_displacement
                    + upper
                ),
                "Prequential_qhat_mm_d": qhat,
                "Calibration_size": int(
                    len(calibration_errors)
                ),
                "Frozen_model_consistency": True,
            }
        ]
    )


# =============================================================================
# Terminal reporting
# =============================================================================

def print_section(
    title: str,
) -> None:
    """Print a formatted terminal section heading."""

    line = "=" * 100

    print(f"\n{line}")
    print(title)
    print(line)


def print_runtime_environment() -> None:
    """Print the principal software environment."""

    print_section(
        "Runtime environment"
    )

    rows = [
        {
            "Component": "Operating system",
            "Version": platform.platform(),
        },
        {
            "Component": "Python",
            "Version": sys.version.split()[0],
        },
        {
            "Component": "NumPy",
            "Version": installed_package_version(
                "numpy"
            ),
        },
        {
            "Component": "pandas",
            "Version": installed_package_version(
                "pandas"
            ),
        },
        {
            "Component": "SciPy",
            "Version": installed_package_version(
                "scipy"
            ),
        },
        {
            "Component": "scikit-learn",
            "Version": installed_package_version(
                "scikit-learn"
            ),
        },
        {
            "Component": "openpyxl",
            "Version": installed_package_version(
                "openpyxl"
            ),
        },
    ]

    print(
        pd.DataFrame(rows).to_string(
            index=False
        )
    )


def print_data_summary(
    data_file: Path,
    raw: pd.DataFrame,
) -> None:
    """Print monitoring-data information."""

    print_section(
        "Monitoring data"
    )

    summary = pd.DataFrame(
        [
            {
                "Property": "Input workbook",
                "Value": str(data_file),
            },
            {
                "Property": "Number of observations",
                "Value": str(len(raw)),
            },
            {
                "Property": "Monitoring period",
                "Value": (
                    f"{raw['Date'].min().date()} to "
                    f"{raw['Date'].max().date()}"
                ),
            },
            {
                "Property": "Rainfall range",
                "Value": (
                    f"{raw['RL'].min():.4f} to "
                    f"{raw['RL'].max():.4f} mm/d"
                ),
            },
            {
                "Property": "Groundwater-depth range",
                "Value": (
                    f"{raw['GL'].min():.4f} to "
                    f"{raw['GL'].max():.4f} m"
                ),
            },
            {
                "Property": "Final cumulative displacement",
                "Value": (
                    f"{raw['SD'].iloc[-1]:.4f} mm"
                ),
            },
            {
                "Property": "Initial latent velocity",
                "Value": (
                    f"{FILTER_CONFIG['initial_velocity']:.6f} mm/d"
                ),
            },
        ]
    )

    print(
        summary.to_string(
            index=False
        )
    )


def print_temporal_summary(
    summaries: list[dict[str, Any]],
) -> None:
    """Print external-block settings and fitted parameters."""

    print_section(
        "External temporal design and selected RSRL-HU settings"
    )

    table = pd.DataFrame(
        summaries
    )

    display_columns = [
        "evaluation",
        "fold",
        "development_start",
        "development_end",
        "development_size",
        "calibration_start",
        "calibration_end",
        "calibration_size",
        "test_start",
        "test_end",
        "test_size",
        "gap_days",
        "residual_configuration",
        "residual_shrinkage",
        "residual_selection_score",
        "groundwater_shrinkage",
        "prior_rho",
        "prior_intercept",
        "groundwater_coefficient",
        "groundwater_activation_fraction",
        "observation_scale",
    ]

    table = table[
        display_columns
    ].copy()

    for column in [
        "development_start",
        "development_end",
        "calibration_start",
        "calibration_end",
        "test_start",
        "test_end",
    ]:
        table[column] = pd.to_datetime(
            table[column]
        ).dt.date

    print(
        table.to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.6f}"
            ),
        )
    )


# =============================================================================
# Command-line interface
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the manuscript-aligned RSRL-HU primary forecasting "
            "workflow without creating output files."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help=(
            "Path to the monitoring-data workbook. The default is "
            "data.xlsx in the script directory."
        ),
    )

    parser.add_argument(
        "--skip-operational",
        action="store_true",
        help=(
            "Complete the manuscript evaluation without generating the "
            "following-day operational forecast."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Main program
# =============================================================================

def main() -> None:
    """Run the complete RSRL-HU primary forecasting workflow."""

    set_random_seed(
        RANDOM_SEED
    )

    arguments = parse_arguments()

    data_file = resolve_data_file(
        arguments.data
    )

    raw = read_monitoring_data(
        data_file
    )

    print_section(
        f"{MODEL_NAME} version {MODEL_VERSION}"
    )

    print(
        "Robust state residual learning informed by hydrology with "
        "prequential uncertainty quantification for one-day-ahead "
        "landslide velocity forecasting"
    )

    print_runtime_environment()

    print_data_summary(
        data_file,
        raw,
    )

    (
        rolling_specifications,
        late_specification,
    ) = construct_temporal_design(
        raw
    )

    rolling_prediction_blocks = []
    fitting_summaries = []

    for specification in rolling_specifications:
        print_section(
            "Fitting rolling-origin block "
            f"{specification['fold']} of "
            f"{len(rolling_specifications)}"
        )

        print(
            "External target period: "
            f"{raw.loc[specification['test_rows'][0], 'Date'].date()} "
            "to "
            f"{raw.loc[specification['test_rows'][-1], 'Date'].date()}"
        )

        predictions, summary = evaluate_external_block(
            raw=raw,
            specification=specification,
            filter_config=FILTER_CONFIG,
            activation_depth=(
                GROUNDWATER_ACTIVATION_DEPTH
            ),
        )

        rolling_prediction_blocks.append(
            predictions
        )

        fitting_summaries.append(
            summary
        )

        print(
            "Selected residual configuration: "
            f"{summary['residual_configuration']}"
        )

        print(
            "Selected residual shrinkage: "
            f"{summary['residual_shrinkage']:.2f}"
        )

        print(
            "Selected groundwater shrinkage: "
            f"{summary['groundwater_shrinkage']:.2f}"
        )

    rolling_predictions = (
        pd.concat(
            rolling_prediction_blocks,
            ignore_index=True,
        )
        .sort_values(
            [
                "Fold",
                "Target_date",
            ]
        )
        .reset_index(drop=True)
    )

    print_section(
        "Fitting prespecified late-period validation"
    )

    print(
        "External target period: "
        f"{raw.loc[late_specification['test_rows'][0], 'Date'].date()} "
        "to "
        f"{raw.loc[late_specification['test_rows'][-1], 'Date'].date()}"
    )

    (
        late_predictions,
        late_summary,
    ) = evaluate_external_block(
        raw=raw,
        specification=late_specification,
        filter_config=FILTER_CONFIG,
        activation_depth=(
            GROUNDWATER_ACTIVATION_DEPTH
        ),
    )

    fitting_summaries.append(
        late_summary
    )

    print(
        "Selected residual configuration: "
        f"{late_summary['residual_configuration']}"
    )

    print(
        "Selected residual shrinkage: "
        f"{late_summary['residual_shrinkage']:.2f}"
    )

    print(
        "Selected groundwater shrinkage: "
        f"{late_summary['groundwater_shrinkage']:.2f}"
    )

    print_temporal_summary(
        fitting_summaries
    )

    rolling_velocity_summary = summarize_velocity_forecasts(
        rolling_predictions,
        "Rolling-origin evaluation",
    )

    late_velocity_summary = summarize_velocity_forecasts(
        late_predictions,
        "Late-period validation",
    )

    velocity_summary = pd.concat(
        [
            rolling_velocity_summary,
            late_velocity_summary,
        ],
        ignore_index=True,
    )

    print_section(
        "RSRL-HU following-day reference latent-velocity performance"
    )

    print(
        velocity_summary.to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.9f}"
            ),
        )
    )

    rolling_groundwater_summary = (
        summarize_groundwater_forecasts(
            rolling_predictions,
            "Rolling-origin evaluation",
        )
    )

    late_groundwater_summary = (
        summarize_groundwater_forecasts(
            late_predictions,
            "Late-period validation",
        )
    )

    groundwater_summary = pd.concat(
        [
            rolling_groundwater_summary,
            late_groundwater_summary,
        ],
        ignore_index=True,
    )

    print_section(
        "Following-day groundwater-depth-change performance"
    )

    print(
        groundwater_summary.to_string(
            index=False,
            float_format=(
                lambda value: f"{value:.8f}"
            ),
        )
    )

    if not arguments.skip_operational:
        matching_configurations = [
            configuration
            for configuration in RESIDUAL_CONFIGURATIONS
            if configuration["configuration"]
            == late_summary[
                "residual_configuration"
            ]
        ]

        if len(matching_configurations) != 1:
            raise RuntimeError(
                "The locked operational residual configuration could not "
                "be identified uniquely."
            )

        locked_parameters = deepcopy(
            matching_configurations[0]
        )

        locked_configuration_id = next(
            index
            for index, configuration in enumerate(
                RESIDUAL_CONFIGURATIONS
            )
            if configuration["configuration"]
            == locked_parameters[
                "configuration"
            ]
        )

        locked_operational_selection = {
            "configuration_id": locked_configuration_id,
            "configuration": locked_parameters[
                "configuration"
            ],
            "parameters": locked_parameters,
            "shrinkage": float(
                late_summary[
                    "residual_shrinkage"
                ]
            ),
            "score": np.nan,
        }

        operational_forecast = generate_operational_forecast(
            raw=raw,
            locked_selection=(
                locked_operational_selection
            ),
            filter_config=FILTER_CONFIG,
            activation_depth=(
                GROUNDWATER_ACTIVATION_DEPTH
            ),
        )

        print_section(
            "Operational following-day RSRL-HU forecast"
        )

        print(
            operational_forecast.to_string(
                index=False,
                float_format=(
                    lambda value: f"{value:.9f}"
                ),
            )
        )

    print_section(
        "Execution completed"
    )

    print(
        "All calculations were completed in memory. "
        "No figures or output files were created."
    )


if __name__ == "__main__":
    main()