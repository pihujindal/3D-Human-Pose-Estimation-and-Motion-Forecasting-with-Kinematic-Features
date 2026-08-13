import os

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# PATHS
# ============================================================

X_TEST_PATH = (
    "data/processed/sequences/X_test.npy"
)

Y_TEST_PATH = (
    "data/processed/sequences/y_test.npy"
)

SCALER_PATH = (
    "data/processed/features/standard_scaler.pkl"
)

OUTPUT_DIR = (
    "data/processed/evaluation/metrics"
)


# ============================================================
# PARAMETERS
# ============================================================

INPUT_WINDOW = 30

PREDICTION_HORIZON = 10

N_FEATURES = 94

N_LANDMARKS = 12

N_COORDINATES = 36


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("LAST-POSE BASELINE")
print("=" * 70)

print()
print("Loading test data...")


X_test = np.load(
    X_TEST_PATH
)

Y_test = np.load(
    Y_TEST_PATH
)


print(
    f"X_test shape : {X_test.shape}"
)

print(
    f"Y_test shape : {Y_test.shape}"
)


# ============================================================
# VALIDATE
# ============================================================

if X_test.ndim != 3:

    raise ValueError(
        "X_test must have 3 dimensions."
    )


if Y_test.ndim != 3:

    raise ValueError(
        "Y_test must have 3 dimensions."
    )


if X_test.shape[1] != INPUT_WINDOW:

    raise ValueError(
        f"Expected {INPUT_WINDOW} input frames."
    )


if X_test.shape[2] != N_FEATURES:

    raise ValueError(
        f"Expected {N_FEATURES} features."
    )


if Y_test.shape[1] != PREDICTION_HORIZON:

    raise ValueError(
        f"Expected {PREDICTION_HORIZON} future frames."
    )


if Y_test.shape[2] != N_FEATURES:

    raise ValueError(
        f"Expected {N_FEATURES} target features."
    )


# ============================================================
# LOAD SCALER
# ============================================================

print()
print("Loading scaler...")

scaler = joblib.load(
    SCALER_PATH
)


# ============================================================
# CREATE BASELINE PREDICTION
# ============================================================

"""
Take the last observed frame.

Shape:

(samples, 94)
"""

last_observed_frame = X_test[
    :, -1, :
]


print(
    "Last observed frame shape:",
    last_observed_frame.shape
)


# ============================================================
# REPEAT LAST FRAME
# ============================================================

"""
Repeat the last frame for all 10 future frames.

Result:

(samples, 10, 94)
"""

baseline_prediction = np.repeat(

    last_observed_frame[
        :,
        np.newaxis,
        :
    ],

    PREDICTION_HORIZON,

    axis=1
)


print(
    "Baseline prediction shape:",
    baseline_prediction.shape
)


# ============================================================
# INVERSE NORMALIZATION
# ============================================================

samples = Y_test.shape[0]

future_frames = Y_test.shape[1]


Y_test_flat = Y_test.reshape(
    -1,
    N_FEATURES
)


baseline_flat = (
    baseline_prediction.reshape(
        -1,
        N_FEATURES
    )
)


Y_test_original = (
    scaler.inverse_transform(
        Y_test_flat
    )
)


baseline_original = (
    scaler.inverse_transform(
        baseline_flat
    )
)


Y_test_original = (
    Y_test_original.reshape(
        samples,
        future_frames,
        N_FEATURES
    )
)


baseline_original = (
    baseline_original.reshape(
        samples,
        future_frames,
        N_FEATURES
    )
)


# ============================================================
# OVERALL METRICS
# ============================================================

actual_flat = (
    Y_test_original.reshape(-1)
)

baseline_flat = (
    baseline_original.reshape(-1)
)


mae = mean_absolute_error(

    actual_flat,

    baseline_flat
)


mse = mean_squared_error(

    actual_flat,

    baseline_flat
)


rmse = np.sqrt(
    mse
)


# ============================================================
# 3D COORDINATES
# ============================================================

actual_coordinates = (
    Y_test_original[
        :,
        :,
        :N_COORDINATES
    ].reshape(
        samples,
        future_frames,
        N_LANDMARKS,
        3
    )
)


baseline_coordinates = (
    baseline_original[
        :,
        :,
        :N_COORDINATES
    ].reshape(
        samples,
        future_frames,
        N_LANDMARKS,
        3
    )
)


# ============================================================
# JOINT POSITION ERROR
# ============================================================

joint_errors = np.linalg.norm(

    actual_coordinates -
    baseline_coordinates,

    axis=-1
)


# ============================================================
# MPJPE
# ============================================================

mpjpe = np.mean(
    joint_errors
)


# ============================================================
# ADE
# ============================================================

frame_errors = np.mean(
    joint_errors,
    axis=2
)


ade = np.mean(
    frame_errors
)


# ============================================================
# FDE
# ============================================================

fde = np.mean(
    frame_errors[:, -1]
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("BASELINE RESULTS")
print("=" * 70)

print(
    f"MAE   : {mae:.6f}"
)

print(
    f"MSE   : {mse:.6f}"
)

print(
    f"RMSE  : {rmse:.6f}"
)

print(
    f"MPJPE : {mpjpe:.6f}"
)

print(
    f"ADE   : {ade:.6f}"
)

print(
    f"FDE   : {fde:.6f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

baseline_results = pd.DataFrame({

    "Model": [
        "Last Pose Baseline"
    ],

    "MAE": [
        mae
    ],

    "MSE": [
        mse
    ],

    "RMSE": [
        rmse
    ],

    "MPJPE": [
        mpjpe
    ],

    "ADE": [
        ade
    ],

    "FDE": [
        fde
    ]
})


output_path = os.path.join(

    OUTPUT_DIR,

    "baseline_results.csv"
)


baseline_results.to_csv(

    output_path,

    index=False
)


# ============================================================
# FUTURE FRAME ERROR
# ============================================================

future_frame_results = []


for frame in range(
    future_frames
):

    frame_mae = np.mean(

        np.abs(

            Y_test_original[
                :,
                frame,
                :
            ]

            -

            baseline_original[
                :,
                frame,
                :
            ]
        )
    )


    future_frame_results.append({

        "Future_Frame":
            frame + 1,

        "MAE":
            frame_mae
    })


future_frame_df = pd.DataFrame(
    future_frame_results
)


future_frame_path = os.path.join(

    OUTPUT_DIR,

    "baseline_future_frame_mae.csv"
)


future_frame_df.to_csv(

    future_frame_path,

    index=False
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("BASELINE COMPLETED")
print("=" * 70)

print(
    "Results saved:"
)

print(
    output_path
)

print(
    future_frame_path
)

print("=" * 70)