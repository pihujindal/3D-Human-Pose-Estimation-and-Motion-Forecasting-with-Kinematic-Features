import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    "models/lstm_motion_forecasting.keras"
)

SCALER_PATH = (
    "data/processed/features/standard_scaler.pkl"
)

X_TEST_PATH = (
    "data/processed/sequences/X_test.npy"
)

Y_TEST_PATH = (
    "data/processed/sequences/y_test.npy"
)

FEATURE_CSV = (
    "data/processed/features/"
    "normalized_pose_features.csv"
)

OUTPUT_DIR = (
    "data/processed/evaluation"
)

PLOT_DIR = (
    "data/processed/evaluation/plots"
)

METRIC_DIR = (
    "data/processed/evaluation/metrics"
)


# ============================================================
# PARAMETERS
# ============================================================

INPUT_WINDOW = 30

PREDICTION_HORIZON = 10

N_FEATURES = 94

N_LANDMARKS = 12

N_COORDINATE_FEATURES = 36


# ============================================================
# LANDMARK NAMES
# ============================================================

LANDMARK_NAMES = [

    "LS",
    "RS",

    "LE",
    "RE",

    "LW",
    "RW",

    "LH",
    "RH",

    "LK",
    "RK",

    "LA",
    "RA"
]


# ============================================================
# SKELETON CONNECTIONS
# ============================================================

SKELETON_CONNECTIONS = [

    # shoulders
    ("LS", "RS"),

    # left arm
    ("LS", "LE"),
    ("LE", "LW"),

    # right arm
    ("RS", "RE"),
    ("RE", "RW"),

    # torso
    ("LS", "LH"),
    ("RS", "RH"),
    ("LH", "RH"),

    # left leg
    ("LH", "LK"),
    ("LK", "LA"),

    # right leg
    ("RH", "RK"),
    ("RK", "RA")
]


LANDMARK_INDEX = {

    name: index

    for index, name in enumerate(
        LANDMARK_NAMES
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
    PLOT_DIR,
    exist_ok=True
)

os.makedirs(
    METRIC_DIR,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("3D HUMAN POSE FORECASTING - EVALUATION")
print("=" * 70)

print()
print("Loading trained model...")

model = load_model(
    MODEL_PATH
)

print(
    "Model loaded successfully."
)


# ============================================================
# LOAD SCALER
# ============================================================

print()
print("Loading scaler...")

scaler = joblib.load(
    SCALER_PATH
)

print(
    "Scaler loaded successfully."
)


# ============================================================
# LOAD TEST DATA
# ============================================================

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
# VALIDATE INPUT
# ============================================================

if X_test.ndim != 3:

    raise ValueError(
        "X_test must be 3-dimensional."
    )


if Y_test.ndim != 3:

    raise ValueError(
        "Y_test must be 3-dimensional."
    )


if X_test.shape[1] != INPUT_WINDOW:

    raise ValueError(
        f"Expected {INPUT_WINDOW} input frames, "
        f"got {X_test.shape[1]}."
    )


if X_test.shape[2] != N_FEATURES:

    raise ValueError(
        f"Expected {N_FEATURES} features, "
        f"got {X_test.shape[2]}."
    )


if Y_test.shape[1] != PREDICTION_HORIZON:

    raise ValueError(
        f"Expected {PREDICTION_HORIZON} "
        f"future frames, "
        f"got {Y_test.shape[1]}."
    )


if Y_test.shape[2] != N_FEATURES:

    raise ValueError(
        f"Expected {N_FEATURES} target features, "
        f"got {Y_test.shape[2]}."
    )


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("GENERATING PREDICTIONS")
print("=" * 70)

Y_pred = model.predict(
    X_test,
    verbose=1
)


print()
print(
    f"Y_pred shape : {Y_pred.shape}"
)


# ============================================================
# CHECK PREDICTION
# ============================================================

if Y_pred.shape != Y_test.shape:

    raise ValueError(
        "\nPrediction shape mismatch!\n"
        f"Y_test = {Y_test.shape}\n"
        f"Y_pred = {Y_pred.shape}"
    )


# ============================================================
# INVERSE NORMALIZATION
# ============================================================

print()
print(
    "Converting normalized values "
    "back to original scale..."
)


samples = Y_test.shape[0]

future_frames = Y_test.shape[1]


# Flatten temporal dimensions

Y_test_flat = Y_test.reshape(
    -1,
    N_FEATURES
)

Y_pred_flat = Y_pred.reshape(
    -1,
    N_FEATURES
)


# Inverse transform

Y_test_original_flat = (
    scaler.inverse_transform(
        Y_test_flat
    )
)

Y_pred_original_flat = (
    scaler.inverse_transform(
        Y_pred_flat
    )
)


# Restore original shape

Y_test_original = (
    Y_test_original_flat.reshape(
        samples,
        future_frames,
        N_FEATURES
    )
)

Y_pred_original = (
    Y_pred_original_flat.reshape(
        samples,
        future_frames,
        N_FEATURES
    )
)


print(
    "Inverse normalization completed."
)


# ============================================================
# LOAD FEATURE NAMES
# ============================================================

print()
print("Loading feature names...")

feature_df = pd.read_csv(
    FEATURE_CSV
)


metadata_columns = [
    "Frame Id",
    "Timestamp"
]


feature_columns = [

    column

    for column in feature_df.columns

    if column not in metadata_columns
]


if len(feature_columns) != N_FEATURES:

    raise ValueError(
        f"Expected {N_FEATURES} feature names, "
        f"found {len(feature_columns)}."
    )


print(
    f"Number of features: "
    f"{len(feature_columns)}"
)


# ============================================================
# 1. OVERALL MAE
# ============================================================

actual_flat = (
    Y_test_original.reshape(-1)
)

predicted_flat = (
    Y_pred_original.reshape(-1)
)


mae = mean_absolute_error(
    actual_flat,
    predicted_flat
)


# ============================================================
# 2. OVERALL MSE
# ============================================================

mse = mean_squared_error(
    actual_flat,
    predicted_flat
)


# ============================================================
# 3. OVERALL RMSE
# ============================================================

rmse = np.sqrt(
    mse
)


print()
print("=" * 70)
print("OVERALL METRICS")
print("=" * 70)

print(
    f"MAE  : {mae:.6f}"
)

print(
    f"MSE  : {mse:.6f}"
)

print(
    f"RMSE : {rmse:.6f}"
)


# ============================================================
# SAVE OVERALL METRICS
# ============================================================

overall_metrics = pd.DataFrame({

    "Metric": [

        "MAE",
        "MSE",
        "RMSE"
    ],

    "Value": [

        mae,
        mse,
        rmse
    ]
})


overall_metrics.to_csv(

    os.path.join(
        METRIC_DIR,
        "overall_metrics.csv"
    ),

    index=False
)


# ============================================================
# 4. FEATURE-WISE MAE
# ============================================================

print()
print("=" * 70)
print("FEATURE-WISE MAE")
print("=" * 70)


feature_mae = np.mean(

    np.abs(
        Y_test_original -
        Y_pred_original
    ),

    axis=(0, 1)
)


feature_results = pd.DataFrame({

    "Feature":
        feature_columns,

    "MAE":
        feature_mae
})


feature_results = (
    feature_results
    .sort_values(
        "MAE"
    )
)


print(
    feature_results.to_string(
        index=False
    )
)


feature_results.to_csv(

    os.path.join(
        METRIC_DIR,
        "feature_wise_mae.csv"
    ),

    index=False
)


# ============================================================
# 5. FEATURE GROUP MAE
# ============================================================

print()
print("=" * 70)
print("FEATURE GROUP MAE")
print("=" * 70)


feature_groups = {

    "3D Coordinates":
        feature_columns[0:36],

    "Body Distances":
        feature_columns[36:46],

    "Joint Angles":
        feature_columns[46:54],

    "Linear Velocity":
        feature_columns[54:66],

    "Linear Acceleration":
        feature_columns[66:78],

    "Angular Velocity":
        feature_columns[78:86],

    "Angular Acceleration":
        feature_columns[86:94]
}


group_results = []


for group_name, group_features in (
    feature_groups.items()
):

    indices = [

        feature_columns.index(
            feature
        )

        for feature in group_features
    ]


    actual_group = (
        Y_test_original[
            :,
            :,
            indices
        ]
    )


    predicted_group = (
        Y_pred_original[
            :,
            :,
            indices
        ]
    )


    group_mae = np.mean(

        np.abs(
            actual_group -
            predicted_group
        )
    )


    group_results.append({

        "Feature_Group":
            group_name,

        "MAE":
            group_mae
    })


    print(
        f"{group_name:25s}"
        f" : {group_mae:.6f}"
    )


group_results_df = pd.DataFrame(
    group_results
)


group_results_df.to_csv(

    os.path.join(
        METRIC_DIR,
        "feature_group_mae.csv"
    ),

    index=False
)


# ============================================================
# 6. MPJPE
# ============================================================

"""
MPJPE = Mean Per Joint Position Error

For every predicted joint:

    error =
        Euclidean distance between
        actual and predicted 3D coordinates

Then average over:

    samples
    future frames
    joints
"""


actual_coordinates = (
    Y_test_original[
        :,
        :,
        :N_COORDINATE_FEATURES
    ].reshape(
        samples,
        future_frames,
        N_LANDMARKS,
        3
    )
)


predicted_coordinates = (
    Y_pred_original[
        :,
        :,
        :N_COORDINATE_FEATURES
    ].reshape(
        samples,
        future_frames,
        N_LANDMARKS,
        3
    )
)


joint_errors = np.linalg.norm(

    actual_coordinates -
    predicted_coordinates,

    axis=-1
)


mpjpe = np.mean(
    joint_errors
)


print()
print(
    "=" * 70
)

print(
    "3D POSE METRICS"
)

print(
    "=" * 70
)

print(
    f"MPJPE : {mpjpe:.6f}"
)


# ============================================================
# 7. ADE
# ============================================================

"""
ADE = Average Displacement Error

Average 3D joint displacement
over the complete predicted trajectory.
"""


frame_position_error = np.mean(
    joint_errors,
    axis=2
)


ade = np.mean(
    frame_position_error
)


print(
    f"ADE   : {ade:.6f}"
)


# ============================================================
# 8. FDE
# ============================================================

"""
FDE = Final Displacement Error

Only the final predicted frame
is considered.
"""


fde = np.mean(
    frame_position_error[:, -1]
)


print(
    f"FDE   : {fde:.6f}"
)


# ============================================================
# SAVE POSE METRICS
# ============================================================

pose_metrics = pd.DataFrame({

    "Metric": [

        "MPJPE",
        "ADE",
        "FDE"
    ],

    "Value": [

        mpjpe,
        ade,
        fde
    ]
})


pose_metrics.to_csv(

    os.path.join(
        METRIC_DIR,
        "pose_metrics.csv"
    ),

    index=False
)


# ============================================================
# 9. FUTURE FRAME MAE
# ============================================================

print()
print("=" * 70)
print("FUTURE FRAME MAE")
print("=" * 70)


future_frame_results = []


for frame in range(
    future_frames
):

    frame_actual = (
        Y_test_original[
            :,
            frame,
            :
        ]
    )


    frame_predicted = (
        Y_pred_original[
            :,
            frame,
            :
        ]
    )


    frame_mae = np.mean(

        np.abs(
            frame_actual -
            frame_predicted
        )
    )


    future_frame_results.append({

        "Future_Frame":
            frame + 1,

        "MAE":
            frame_mae
    })


    print(
        f"Frame {frame + 1:2d}"
        f" : {frame_mae:.6f}"
    )


future_frame_df = pd.DataFrame(
    future_frame_results
)


future_frame_df.to_csv(

    os.path.join(
        METRIC_DIR,
        "future_frame_mae.csv"
    ),

    index=False
)


# ============================================================
# 10. ACTUAL VS PREDICTED FEATURE GRAPHS
# ============================================================

print()
print("=" * 70)
print("GENERATING FEATURE GRAPHS")
print("=" * 70)


# Select representative features
# from different feature groups.

selected_features = [

    "LS_X",

    "LS_Y",

    "LS_Z",

    "Shoulder_Width",

    "Left_Knee_Angle",

    "LS_Velocity",

    "LK_Acceleration",

    "Left_Knee_Angular_Velocity",

    "Left_Knee_Angular_Acceleration"
]


for feature_name in selected_features:

    if feature_name not in feature_columns:

        print(
            f"Skipping missing feature: "
            f"{feature_name}"
        )

        continue


    feature_index = (
        feature_columns.index(
            feature_name
        )
    )


    # --------------------------------------------------------
    # Select FIRST test sample
    # --------------------------------------------------------

    actual_values = (
        Y_test_original[
            0,
            :,
            feature_index
        ]
    )


    predicted_values = (
        Y_pred_original[
            0,
            :,
            feature_index
        ]
    )


    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )


    plt.plot(
        range(1, future_frames + 1),
        actual_values,
        marker="o",
        linewidth=2,
        label="Actual"
    )


    plt.plot(
        range(1, future_frames + 1),
        predicted_values,
        marker="x",
        linewidth=2,
        linestyle="--",
        label="Predicted"
    )


    plt.xlabel(
        "Future Frame"
    )


    plt.ylabel(
        feature_name
    )


    plt.title(
        f"Actual vs Predicted - "
        f"{feature_name}"
    )


    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )


    plt.tight_layout()


    filename = (
        "actual_vs_predicted_"
        + feature_name
        + ".png"
    )


    output_path = os.path.join(
        PLOT_DIR,
        filename
    )


    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )


    print(
        f"Saved: {output_path}"
    )


    plt.show()

    plt.close()


# ============================================================
# 11. FEATURE GROUP BAR GRAPH
# ============================================================

plt.figure(
    figsize=(11, 6)
)


plt.bar(

    group_results_df[
        "Feature_Group"
    ],

    group_results_df[
        "MAE"
    ]
)


plt.xlabel(
    "Feature Group"
)

plt.ylabel(
    "MAE"
)

plt.title(
    "MAE by Feature Group"
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


group_plot_path = os.path.join(

    PLOT_DIR,

    "feature_group_mae.png"
)


plt.savefig(
    group_plot_path,
    dpi=200,
    bbox_inches="tight"
)


print(
    f"Saved: {group_plot_path}"
)


plt.show()

plt.close()


# ============================================================
# 12. FUTURE FRAME ERROR GRAPH
# ============================================================

plt.figure(
    figsize=(9, 5)
)


plt.plot(

    future_frame_df[
        "Future_Frame"
    ],

    future_frame_df[
        "MAE"
    ],

    marker="o",

    linewidth=2
)


plt.xlabel(
    "Future Frame"
)

plt.ylabel(
    "MAE"
)

plt.title(
    "Prediction Error Across Future Frames"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


future_plot_path = os.path.join(

    PLOT_DIR,

    "future_frame_mae.png"
)


plt.savefig(
    future_plot_path,
    dpi=200,
    bbox_inches="tight"
)


print(
    f"Saved: {future_plot_path}"
)


plt.show()

plt.close()


# ============================================================
# 13. 3D ACTUAL VS PREDICTED
# ============================================================

print()
print("=" * 70)
print("GENERATING 3D POSE COMPARISONS")
print("=" * 70)


# Use first test sample
sample_index = 0


actual_sample = (
    actual_coordinates[
        sample_index
    ]
)


predicted_sample = (
    predicted_coordinates[
        sample_index
    ]
)


for frame in range(
    future_frames
):

    actual_pose = (
        actual_sample[
            frame
        ]
    )


    predicted_pose = (
        predicted_sample[
            frame
        ]
    )


    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(14, 7)
    )


    # ========================================================
    # ACTUAL
    # ========================================================

    ax1 = fig.add_subplot(
        121,
        projection="3d"
    )


    ax1.set_title(
        f"Actual Pose\n"
        f"Future Frame {frame + 1}"
    )


    for index, name in enumerate(
        LANDMARK_NAMES
    ):

        x = actual_pose[
            index,
            0
        ]

        y = actual_pose[
            index,
            1
        ]

        z = actual_pose[
            index,
            2
        ]


        ax1.scatter(
            x,
            y,
            z,
            s=60
        )


        ax1.text(
            x,
            y,
            z,
            name,
            fontsize=8
        )


    for start, end in (
        SKELETON_CONNECTIONS
    ):

        i = LANDMARK_INDEX[
            start
        ]

        j = LANDMARK_INDEX[
            end
        ]


        ax1.plot(

            [
                actual_pose[
                    i,
                    0
                ],

                actual_pose[
                    j,
                    0
                ]
            ],

            [
                actual_pose[
                    i,
                    1
                ],

                actual_pose[
                    j,
                    1
                ]
            ],

            [
                actual_pose[
                    i,
                    2
                ],

                actual_pose[
                    j,
                    2
                ]
            ],

            linewidth=2
        )


    ax1.set_xlabel("X")

    ax1.set_ylabel("Y")

    ax1.set_zlabel("Z")


    # ========================================================
    # PREDICTED
    # ========================================================

    ax2 = fig.add_subplot(
        122,
        projection="3d"
    )


    ax2.set_title(
        f"LSTM Predicted Pose\n"
        f"Future Frame {frame + 1}"
    )


    for index, name in enumerate(
        LANDMARK_NAMES
    ):

        x = predicted_pose[
            index,
            0
        ]

        y = predicted_pose[
            index,
            1
        ]

        z = predicted_pose[
            index,
            2
        ]


        ax2.scatter(
            x,
            y,
            z,
            s=60
        )


        ax2.text(
            x,
            y,
            z,
            name,
            fontsize=8
        )


    for start, end in (
        SKELETON_CONNECTIONS
    ):

        i = LANDMARK_INDEX[
            start
        ]

        j = LANDMARK_INDEX[
            end
        ]


        ax2.plot(

            [
                predicted_pose[
                    i,
                    0
                ],

                predicted_pose[
                    j,
                    0
                ]
            ],

            [
                predicted_pose[
                    i,
                    1
                ],

                predicted_pose[
                    j,
                    1
                ]
            ],

            [
                predicted_pose[
                    i,
                    2
                ],

                predicted_pose[
                    j,
                    2
                ]
            ],

            linewidth=2
        )


    ax2.set_xlabel("X")

    ax2.set_ylabel("Y")

    ax2.set_zlabel("Z")


    # ========================================================
    # SAME AXIS LIMITS
    # ========================================================

    combined = np.vstack(
        [
            actual_pose,
            predicted_pose
        ]
    )


    x_min = combined[:, 0].min()
    x_max = combined[:, 0].max()

    y_min = combined[:, 1].min()
    y_max = combined[:, 1].max()

    z_min = combined[:, 2].min()
    z_max = combined[:, 2].max()


    x_pad = max(
        (x_max - x_min) * 0.2,
        0.05
    )

    y_pad = max(
        (y_max - y_min) * 0.2,
        0.05
    )

    z_pad = max(
        (z_max - z_min) * 0.2,
        0.05
    )


    for axis in [
        ax1,
        ax2
    ]:

        axis.set_xlim(
            x_min - x_pad,
            x_max + x_pad
        )

        axis.set_ylim(
            y_min - y_pad,
            y_max + y_pad
        )

        axis.set_zlim(
            z_min - z_pad,
            z_max + z_pad
        )


    # ========================================================
    # SAVE
    # ========================================================

    plt.tight_layout()


    output_path = os.path.join(

        PLOT_DIR,

        f"3d_actual_vs_predicted_"
        f"frame_{frame + 1}.png"
    )


    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )


    print(
        f"Saved: {output_path}"
    )


    # ========================================================
    # SHOW
    # ========================================================

    plt.show()

    plt.close()


# ============================================================
# 14. SAVE ALL ACTUAL VS PREDICTED VALUES
# ============================================================

print()
print(
    "Saving complete prediction table..."
)


prediction_rows = []


for sample in range(
    samples
):

    for frame in range(
        future_frames
    ):

        row = {

            "Sample":
                sample,

            "Future_Frame":
                frame + 1
        }


        for feature_index, feature_name in enumerate(
            feature_columns
        ):

            row[
                "Actual_" + feature_name
            ] = Y_test_original[
                sample,
                frame,
                feature_index
            ]


            row[
                "Predicted_" + feature_name
            ] = Y_pred_original[
                sample,
                frame,
                feature_index
            ]


        prediction_rows.append(
            row
        )


prediction_df = pd.DataFrame(
    prediction_rows
)


prediction_csv_path = os.path.join(

    OUTPUT_DIR,

    "actual_vs_predicted_all.csv"
)


prediction_df.to_csv(

    prediction_csv_path,

    index=False
)


print(
    f"Saved: {prediction_csv_path}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)

print()
print(
    "Overall Metrics"
)

print(
    f"MAE   = {mae:.6f}"
)

print(
    f"MSE   = {mse:.6f}"
)

print(
    f"RMSE  = {rmse:.6f}"
)

print(
    f"MPJPE = {mpjpe:.6f}"
)

print(
    f"ADE   = {ade:.6f}"
)

print(
    f"FDE   = {fde:.6f}"
)

print()
print(
    "Results directory:"
)

print(
    OUTPUT_DIR
)

print()
print(
    "Evaluation finished successfully."
)

print("=" * 70)