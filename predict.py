import os

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

SEQUENCE_DIR = (
    "data/processed/sequences"
)

MODEL_PATH = (
    "models/lstm_motion_forecasting.keras"
)

SCALER_PATH = (
    "data/processed/features/standard_scaler.pkl"
)

FEATURE_CSV = (
    "data/processed/features/normalized_pose_features.csv"
)

OUTPUT_DIR = (
    "data/processed/predictions"
)


# ============================================================
# PARAMETERS
# ============================================================

INPUT_WINDOW = 30

PREDICTION_HORIZON = 10

N_FEATURES = 94


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD TEST INPUT
# ============================================================

print()
print("=" * 70)
print("3D HUMAN MOTION FORECASTING")
print("PREDICTION")
print("=" * 70)

print()
print("Loading test input...")


X_test = np.load(
    os.path.join(
        SEQUENCE_DIR,
        "X_test.npy"
    )
)


print(
    f"X_test shape: {X_test.shape}"
)


# ============================================================
# VALIDATE INPUT
# ============================================================

if X_test.ndim != 3:

    raise ValueError(
        f"X_test must be 3-dimensional. "
        f"Got {X_test.shape}"
    )


if X_test.shape[1] != INPUT_WINDOW:

    raise ValueError(

        f"Expected input window "
        f"{INPUT_WINDOW}."

        f"Got {X_test.shape[1]}"
    )


if X_test.shape[2] != N_FEATURES:

    raise ValueError(

        f"Expected {N_FEATURES} features."

        f"Got {X_test.shape[2]}"
    )


if np.isnan(X_test).any():

    raise ValueError(
        "NaN values found in X_test."
    )


if not np.isfinite(X_test).all():

    raise ValueError(
        "Infinite values found in X_test."
    )


# ============================================================
# LOAD MODEL
# ============================================================

print()
print(
    "Loading trained LSTM model..."
)


model = tf.keras.models.load_model(
    MODEL_PATH
)


print(
    "Model loaded successfully."
)


# ============================================================
# MODEL OUTPUT CHECK
# ============================================================

print()
print(
    "Checking model output..."
)


test_output = model.predict(

    X_test[:1],

    verbose=0
)


print(
    f"Model output shape: "
    f"{test_output.shape}"
)


expected_shape = (

    1,

    PREDICTION_HORIZON,

    N_FEATURES
)


if test_output.shape != expected_shape:

    raise ValueError(

        "\nUnexpected model output shape.\n"

        f"Expected: {expected_shape}\n"

        f"Received: {test_output.shape}"
    )


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("GENERATING FUTURE POSE PREDICTIONS")
print("=" * 70)


predictions_normalized = model.predict(

    X_test,

    verbose=1
)


print()
print(
    f"Prediction shape: "
    f"{predictions_normalized.shape}"
)


# ============================================================
# SAVE NORMALIZED PREDICTIONS
# ============================================================

normalized_path = os.path.join(

    OUTPUT_DIR,

    "predictions_normalized.npy"
)


np.save(

    normalized_path,

    predictions_normalized
)


print()
print(
    "Normalized predictions saved:"
)

print(
    normalized_path
)


# ============================================================
# LOAD SCALER
# ============================================================

print()
print(
    "Loading feature scaler..."
)


if not os.path.exists(
    SCALER_PATH
):

    raise FileNotFoundError(

        "\nScaler not found:\n"

        f"{SCALER_PATH}\n\n"

        "Make sure normalization.py "
        "saved the scaler."
    )


scaler = joblib.load(
    SCALER_PATH
)


print(
    "Scaler loaded successfully."
)


# ============================================================
# INVERSE NORMALIZATION
# ============================================================

print()
print(
    "Converting predictions "
    "to original feature scale..."
)


samples = predictions_normalized.shape[0]


prediction_flat = (
    predictions_normalized.reshape(
        -1,
        N_FEATURES
    )
)


predictions_original_flat = (
    scaler.inverse_transform(
        prediction_flat
    )
)


predictions_original = (
    predictions_original_flat.reshape(
        samples,
        PREDICTION_HORIZON,
        N_FEATURES
    )
)


print(
    "Inverse normalization completed."
)


# ============================================================
# SAVE ORIGINAL-SCALE PREDICTIONS
# ============================================================

original_path = os.path.join(

    OUTPUT_DIR,

    "predictions_original.npy"
)


np.save(

    original_path,

    predictions_original
)


print()
print(
    "Original-scale predictions saved:"
)

print(
    original_path
)


# ============================================================
# LOAD FEATURE NAMES
# ============================================================

print()
print(
    "Loading feature names..."
)


if not os.path.exists(
    FEATURE_CSV
):

    raise FileNotFoundError(

        f"Feature CSV not found:\n"
        f"{FEATURE_CSV}"
    )


feature_df = pd.read_csv(
    FEATURE_CSV
)


metadata_columns = [

    "Frame Id",
    "Frame_ID",
    "frame_id",
    "Frame",
    "Timestamp",
    "timestamp",
    "Time",
    "time"
]


feature_columns = [

    column

    for column in feature_df.columns

    if column not in metadata_columns
]


if len(feature_columns) != N_FEATURES:

    raise ValueError(

        f"Expected {N_FEATURES} "
        f"feature columns.\n"

        f"Found {len(feature_columns)}."
    )


# ============================================================
# SAVE FIRST SAMPLE AS CSV
# ============================================================

print()
print(
    "Saving first test prediction..."
)


first_sample = predictions_original[0]


rows = []


for frame_index in range(
    PREDICTION_HORIZON
):

    row = {

        "Future_Frame":
            frame_index + 1
    }


    for feature_index, feature_name in enumerate(
        feature_columns
    ):

        row[
            feature_name
        ] = first_sample[
            frame_index,
            feature_index
        ]


    rows.append(
        row
    )


first_prediction_df = pd.DataFrame(
    rows
)


first_prediction_path = os.path.join(

    OUTPUT_DIR,

    "first_sample_prediction.csv"
)


first_prediction_df.to_csv(

    first_prediction_path,

    index=False
)


print(
    "First sample prediction saved:"
)

print(
    first_prediction_path
)


# ============================================================
# SAVE ALL PREDICTIONS AS CSV
# ============================================================

print()
print(
    "Saving all predictions to CSV..."
)


all_rows = []


for sample_index in range(
    samples
):

    for frame_index in range(
        PREDICTION_HORIZON
    ):

        row = {

            "Sample":
                sample_index,

            "Future_Frame":
                frame_index + 1
        }


        for feature_index, feature_name in enumerate(
            feature_columns
        ):

            row[
                feature_name
            ] = predictions_original[
                sample_index,
                frame_index,
                feature_index
            ]


        all_rows.append(
            row
        )


all_predictions_df = pd.DataFrame(
    all_rows
)


all_prediction_path = os.path.join(

    OUTPUT_DIR,

    "all_predictions.csv"
)


all_predictions_df.to_csv(

    all_prediction_path,

    index=False
)


print(
    "All predictions saved:"
)

print(
    all_prediction_path
)


# ============================================================
# DISPLAY SAMPLE PREDICTION
# ============================================================

print()
print("=" * 70)
print("FIRST TEST SAMPLE")
print("=" * 70)


for frame_index in range(
    PREDICTION_HORIZON
):

    print()
    print(
        f"Future Frame "
        f"{frame_index + 1}"
    )

    print(
        first_prediction_df.iloc[
            frame_index
        ].to_string()
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("PREDICTION COMPLETED")
print("=" * 70)

print()
print(
    "Normalized predictions:"
)

print(
    normalized_path
)

print()
print(
    "Original-scale predictions:"
)

print(
    original_path
)

print()
print(
    "First sample CSV:"
)

print(
    first_prediction_path
)

print()
print(
    "All predictions CSV:"
)

print(
    all_prediction_path
)

print()
print(
    f"Total test samples: {samples}"
)

print(
    f"Future frames per sample: "
    f"{PREDICTION_HORIZON}"
)

print(
    f"Features per frame: "
    f"{N_FEATURES}"
)

print("=" * 70)