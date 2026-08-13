import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# PATHS
# ============================================================

X_TRAIN_PATH = (
    "data/processed/sequences/X_train.npy"
)

Y_TRAIN_PATH = (
    "data/processed/sequences/y_train.npy"
)

X_VAL_PATH = (
    "data/processed/sequences/X_val.npy"
)

Y_VAL_PATH = (
    "data/processed/sequences/y_val.npy"
)

X_TEST_PATH = (
    "data/processed/sequences/X_test.npy"
)

Y_TEST_PATH = (
    "data/processed/sequences/y_test.npy"
)


OUTPUT_DIR = (
    "data/processed/evaluation/ablation"
)

MODEL_DIR = (
    "data/processed/evaluation/ablation/models"
)

PLOT_DIR = (
    "data/processed/evaluation/ablation/plots"
)


# ============================================================
# PARAMETERS
# ============================================================

INPUT_WINDOW = 30

PREDICTION_HORIZON = 10

TOTAL_FEATURES = 94

N_LANDMARKS = 12

COORDINATE_FEATURES = 36


EPOCHS = 50

BATCH_SIZE = 32

LEARNING_RATE = 0.001


# ============================================================
# FEATURE RANGES
# ============================================================

"""
Feature layout:

0   - 35  -> Coordinates
36  - 45  -> Body Distances
46  - 53  -> Joint Angles
54  - 65  -> Linear Velocity
66  - 77  -> Linear Acceleration
78  - 85  -> Angular Velocity
86  - 93  -> Angular Acceleration
"""


FEATURE_CONFIGURATIONS = {

    "Coordinates": list(
        range(0, 36)
    ),

    "Coordinates_Distances": list(
        range(0, 46)
    ),

    "Coordinates_Distances_Angles": list(
        range(0, 54)
    ),

    "Coordinates_Linear_Kinematics": list(
        range(0, 78)
    ),

    "All_94_Features": list(
        range(0, 94)
    )
}


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    PLOT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FEATURE ABLATION STUDY")
print("=" * 70)

print()
print("Loading datasets...")


X_train = np.load(
    X_TRAIN_PATH
)

Y_train = np.load(
    Y_TRAIN_PATH
)

X_val = np.load(
    X_VAL_PATH
)

Y_val = np.load(
    Y_VAL_PATH
)

X_test = np.load(
    X_TEST_PATH
)

Y_test = np.load(
    Y_TEST_PATH
)


print(
    "X_train:",
    X_train.shape
)

print(
    "Y_train:",
    Y_train.shape
)

print(
    "X_val:",
    X_val.shape
)

print(
    "Y_val:",
    Y_val.shape
)

print(
    "X_test:",
    X_test.shape
)

print(
    "Y_test:",
    Y_test.shape
)


# ============================================================
# VALIDATE DATA
# ============================================================

if X_train.shape[2] != TOTAL_FEATURES:

    raise ValueError(
        "X_train does not contain "
        "94 features."
    )


if Y_train.shape[2] != TOTAL_FEATURES:

    raise ValueError(
        "Y_train does not contain "
        "94 features."
    )


if X_val.shape[2] != TOTAL_FEATURES:

    raise ValueError(
        "X_val does not contain "
        "94 features."
    )


if Y_val.shape[2] != TOTAL_FEATURES:

    raise ValueError(
        "Y_val does not contain "
        "94 features."
    )


if X_test.shape[2] != TOTAL_FEATURES:

    raise ValueError(
        "X_test does not contain "
        "94 features."
    )


if Y_test.shape[2] != TOTAL_FEATURES:

    raise ValueError(
        "Y_test does not contain "
        "94 features."
    )


# ============================================================
# MODEL FUNCTION
# ============================================================

def build_model(
    input_features
):
    """
    Build the same LSTM architecture
    for every ablation experiment.

    Only the number of input features
    changes.
    """

    model = Sequential()


    model.add(
        LSTM(
            128,
            return_sequences=True,
            input_shape=(
                INPUT_WINDOW,
                input_features
            )
        )
    )


    model.add(
        Dropout(
            0.2
        )
    )


    model.add(
        LSTM(
            64
        )
    )


    model.add(
        Dropout(
            0.2
        )
    )


    model.add(
        Dense(
            128,
            activation="relu"
        )
    )


    model.add(
        Dense(
            PREDICTION_HORIZON
            * input_features
        )
    )


    model.add(
        Dense(
            PREDICTION_HORIZON
            * input_features
        )
    )


    model.compile(

        optimizer="adam",

        loss="mse",

        metrics=[
            "mae"
        ]
    )


    return model


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    actual,
    predicted,
    coordinate_features
):
    """
    Calculate forecasting metrics.

    actual:
        (samples, future_frames, features)

    predicted:
        (samples, future_frames, features)
    """

    actual_flat = (
        actual.reshape(-1)
    )

    predicted_flat = (
        predicted.reshape(-1)
    )


    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = mean_absolute_error(
        actual_flat,
        predicted_flat
    )


    # --------------------------------------------------------
    # MSE
    # --------------------------------------------------------

    mse = mean_squared_error(
        actual_flat,
        predicted_flat
    )


    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = np.sqrt(
        mse
    )


    # --------------------------------------------------------
    # 3D coordinates
    # --------------------------------------------------------

    actual_pose = (
        actual[
            :,
            :,
            :coordinate_features
        ]
    )


    predicted_pose = (
        predicted[
            :,
            :,
            :coordinate_features
        ]
    )


    # Number of joints

    number_of_joints = (
        coordinate_features // 3
    )


    actual_pose = (
        actual_pose.reshape(
            actual.shape[0],
            actual.shape[1],
            number_of_joints,
            3
        )
    )


    predicted_pose = (
        predicted_pose.reshape(
            predicted.shape[0],
            predicted.shape[1],
            number_of_joints,
            3
        )
    )


    # --------------------------------------------------------
    # Euclidean joint error
    # --------------------------------------------------------

    joint_error = np.linalg.norm(

        actual_pose -
        predicted_pose,

        axis=-1
    )


    # --------------------------------------------------------
    # MPJPE
    # --------------------------------------------------------

    mpjpe = np.mean(
        joint_error
    )


    # --------------------------------------------------------
    # ADE
    # --------------------------------------------------------

    frame_error = np.mean(
        joint_error,
        axis=2
    )


    ade = np.mean(
        frame_error
    )


    # --------------------------------------------------------
    # FDE
    # --------------------------------------------------------

    fde = np.mean(
        frame_error[:, -1]
    )


    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MPJPE": mpjpe,
        "ADE": ade,
        "FDE": fde
    }


# ============================================================
# RUN ABLATION EXPERIMENTS
# ============================================================

results = []


for experiment_name, feature_indices in (
    FEATURE_CONFIGURATIONS.items()
):

    print()
    print()
    print("=" * 70)

    print(
        f"EXPERIMENT: "
        f"{experiment_name}"
    )

    print("=" * 70)


    number_of_features = len(
        feature_indices
    )


    print(
        f"Number of features: "
        f"{number_of_features}"
    )


    # ========================================================
    # SELECT FEATURES
    # ========================================================

    X_train_subset = (
        X_train[
            :,
            :,
            feature_indices
        ]
    )


    X_val_subset = (
        X_val[
            :,
            :,
            feature_indices
        ]
    )


    X_test_subset = (
        X_test[
            :,
            :,
            feature_indices
        ]
    )


    Y_train_subset = (
        Y_train[
            :,
            :,
            feature_indices
        ]
    )


    Y_val_subset = (
        Y_val[
            :,
            :,
            feature_indices
        ]
    )


    Y_test_subset = (
        Y_test[
            :,
            :,
            feature_indices
        ]
    )


    # ========================================================
    # BUILD MODEL
    # ========================================================

    print()
    print(
        "Building model..."
    )


    model = build_model(
        number_of_features
    )


    model.summary()


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    early_stopping = EarlyStopping(

        monitor="val_loss",

        patience=8,

        restore_best_weights=True
    )


    # ========================================================
    # TRAIN
    # ========================================================

    print()
    print(
        "Training..."
    )


    history = model.fit(

        X_train_subset,

        Y_train_subset.reshape(
            Y_train_subset.shape[0],
            -1
        ),

        validation_data=(

            X_val_subset,

            Y_val_subset.reshape(
                Y_val_subset.shape[0],
                -1
            )
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        callbacks=[
            early_stopping
        ],

        verbose=1
    )


    # ========================================================
    # PREDICT
    # ========================================================

    print()
    print(
        "Generating predictions..."
    )


    prediction_flat = model.predict(

        X_test_subset,

        verbose=0
    )


    prediction = (
        prediction_flat.reshape(
            Y_test_subset.shape
        )
    )


    # ========================================================
    # METRICS
    # ========================================================

    metrics = calculate_metrics(

        Y_test_subset,

        prediction,

        min(
            COORDINATE_FEATURES,
            number_of_features
        )
    )


    # ========================================================
    # STORE RESULTS
    # ========================================================

    result = {

        "Experiment":
            experiment_name,

        "Number_of_Features":
            number_of_features,

        "MAE":
            metrics["MAE"],

        "MSE":
            metrics["MSE"],

        "RMSE":
            metrics["RMSE"],

        "MPJPE":
            metrics["MPJPE"],

        "ADE":
            metrics["ADE"],

        "FDE":
            metrics["FDE"],

        "Best_Validation_Loss":
            min(
                history.history[
                    "val_loss"
                ]
            ),

        "Epochs_Trained":
            len(
                history.history[
                    "loss"
                ]
            )
    }


    results.append(
        result
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = os.path.join(

        MODEL_DIR,

        f"{experiment_name}.keras"
    )


    model.save(
        model_path
    )


    print()
    print(
        "Model saved:"
    )

    print(
        model_path
    )


    # ========================================================
    # SAVE TRAINING CURVE
    # ========================================================

    plt.figure(
        figsize=(9, 5)
    )


    plt.plot(

        history.history[
            "loss"
        ],

        label="Training Loss"
    )


    plt.plot(

        history.history[
            "val_loss"
        ],

        label="Validation Loss"
    )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "MSE Loss"
    )


    plt.title(
        f"Training Curve - "
        f"{experiment_name}"
    )


    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()


    curve_path = os.path.join(

        PLOT_DIR,

        f"{experiment_name}_training.png"
    )


    plt.savefig(

        curve_path,

        dpi=200,

        bbox_inches="tight"
    )


    plt.close()


    # ========================================================
    # PRINT METRICS
    # ========================================================

    print()
    print(
        "Experiment Results"
    )

    print(
        f"MAE   : "
        f"{metrics['MAE']:.6f}"
    )

    print(
        f"RMSE  : "
        f"{metrics['RMSE']:.6f}"
    )

    print(
        f"MPJPE : "
        f"{metrics['MPJPE']:.6f}"
    )

    print(
        f"ADE   : "
        f"{metrics['ADE']:.6f}"
    )

    print(
        f"FDE   : "
        f"{metrics['FDE']:.6f}"
    )


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SORT BY MPJPE
# ============================================================

results_df = (
    results_df
    .sort_values(
        "MPJPE"
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = os.path.join(

    OUTPUT_DIR,

    "ablation_results.csv"
)


results_df.to_csv(

    results_path,

    index=False
)


# ============================================================
# PRINT FINAL TABLE
# ============================================================

print()
print()
print("=" * 70)
print("FINAL ABLATION RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# CREATE COMPARISON GRAPH
# ============================================================

plot_metrics = [

    "MAE",
    "RMSE",
    "MPJPE",
    "ADE",
    "FDE"
]


for metric in plot_metrics:

    plt.figure(
        figsize=(11, 6)
    )


    plt.bar(

        results_df[
            "Experiment"
        ],

        results_df[
            metric
        ]
    )


    plt.xlabel(
        "Feature Configuration"
    )


    plt.ylabel(
        metric
    )


    plt.title(
        f"Ablation Study - "
        f"{metric}"
    )


    plt.xticks(
        rotation=30,
        ha="right"
    )


    plt.grid(
        axis="y",
        alpha=0.3
    )


    plt.tight_layout()


    plot_path = os.path.join(

        PLOT_DIR,

        f"ablation_{metric.lower()}.png"
    )


    plt.savefig(

        plot_path,

        dpi=200,

        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("ABLATION STUDY COMPLETED")
print("=" * 70)

print()
print(
    "Results:"
)

print(
    results_path
)

print()
print(
    "Models:"
)

print(
    MODEL_DIR
)

print()
print(
    "Plots:"
)

print(
    PLOT_DIR
)

print("=" * 70)