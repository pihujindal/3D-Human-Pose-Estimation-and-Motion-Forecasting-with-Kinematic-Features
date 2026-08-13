import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = "data/processed"

EVALUATION_DIR = os.path.join(
    BASE_DIR,
    "evaluation"
)

METRICS_DIR = os.path.join(
    EVALUATION_DIR,
    "metrics"
)

PLOTS_DIR = os.path.join(
    EVALUATION_DIR,
    "plots"
)

BASELINE_DIR = os.path.join(
    EVALUATION_DIR,
    "baseline"
)

ABLATION_DIR = os.path.join(
    EVALUATION_DIR,
    "ablation"
)

PREDICTION_DIR = os.path.join(
    BASE_DIR,
    "predictions"
)

MODEL_DIR = "models"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    PLOTS_DIR,
    exist_ok=True
)


# ============================================================
# PARAMETERS
# ============================================================

PREDICTION_HORIZON = 10

N_LANDMARKS = 12

COORDINATES_PER_LANDMARK = 3

TARGET_FEATURES = 36


# ============================================================
# HELPER FUNCTION
# ============================================================

def save_and_show(
    filename
):
    """
    Save current matplotlib figure and show it.
    """

    output_path = os.path.join(
        PLOTS_DIR,
        filename
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 80)
print("RESEARCH VISUALIZATION")
print("3D HUMAN POSE ESTIMATION AND MOTION FORECASTING")
print("=" * 80)


# ============================================================
# 1. TRAINING HISTORY
# ============================================================

print()
print("=" * 80)
print("1. TRAINING HISTORY")
print("=" * 80)


history_path = os.path.join(
    MODEL_DIR,
    "training_history.npy"
)


if os.path.exists(
    history_path
):

    history = np.load(
        history_path,
        allow_pickle=True
    ).item()


    train_loss = history.get(
        "loss"
    )

    val_loss = history.get(
        "val_loss"
    )


    if (
        train_loss is not None
        and val_loss is not None
    ):

        epochs = np.arange(
            1,
            len(train_loss) + 1
        )


        plt.figure(
            figsize=(10, 6)
        )


        plt.plot(

            epochs,

            train_loss,

            marker="o",

            label="Training Loss"
        )


        plt.plot(

            epochs,

            val_loss,

            marker="o",

            label="Validation Loss"
        )


        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "MSE Loss"
        )

        plt.title(
            "Training and Validation Loss"
        )

        plt.legend()

        plt.grid(
            True,
            alpha=0.3
        )


        save_and_show(
            "training_validation_loss.png"
        )


    else:

        print(
            "Training/validation loss not available."
        )

else:

    print(
        f"History file not found: "
        f"{history_path}"
    )


# ============================================================
# 2. FUTURE FRAME MAE
# ============================================================

print()
print("=" * 80)
print("2. FUTURE FRAME MAE")
print("=" * 80)


future_metrics_path = os.path.join(

    METRICS_DIR,

    "future_frame_metrics.csv"
)


if os.path.exists(
    future_metrics_path
):

    future_df = pd.read_csv(
        future_metrics_path
    )


    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(

        future_df[
            "Future_Frame"
        ],

        future_df[
            "MAE"
        ],

        marker="o"
    )


    plt.xlabel(
        "Future Frame"
    )

    plt.ylabel(
        "MAE"
    )

    plt.title(
        "MAE Across Future Prediction Horizon"
    )

    plt.grid(
        True,
        alpha=0.3
    )


    save_and_show(
        "research_future_frame_mae.png"
    )

else:

    print(
        "Future-frame metrics not found."
    )


# ============================================================
# 3. FUTURE FRAME MPJPE
# ============================================================

print()
print("=" * 80)
print("3. FUTURE FRAME MPJPE")
print("=" * 80)


if os.path.exists(
    future_metrics_path
):

    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(

        future_df[
            "Future_Frame"
        ],

        future_df[
            "MPJPE"
        ],

        marker="o"
    )


    plt.xlabel(
        "Future Frame"
    )

    plt.ylabel(
        "MPJPE"
    )

    plt.title(
        "MPJPE Across Future Prediction Horizon"
    )

    plt.grid(
        True,
        alpha=0.3
    )


    save_and_show(
        "research_future_frame_mpjpe.png"
    )


# ============================================================
# LOAD PREDICTIONS
# ============================================================

print()
print("=" * 80)
print("LOADING PREDICTIONS")
print("=" * 80)


prediction_path = os.path.join(

    PREDICTION_DIR,

    "predictions_original.npy"
)


ground_truth_path = os.path.join(

    PREDICTION_DIR,

    "ground_truth_original.npy"
)


if (
    not os.path.exists(
        prediction_path
    )

    or

    not os.path.exists(
        ground_truth_path
    )
):

    print(
        "Prediction files not found."
    )

    predictions = None

    ground_truth = None

else:

    predictions = np.load(
        prediction_path
    )

    ground_truth = np.load(
        ground_truth_path
    )


    print(
        f"Predictions: "
        f"{predictions.shape}"
    )

    print(
        f"Ground truth: "
        f"{ground_truth.shape}"
    )


# ============================================================
# FIRST SAMPLE VISUALIZATION
# ============================================================

if (
    predictions is not None
    and ground_truth is not None
):

    sample_index = 0

    sample_prediction = (
        predictions[
            sample_index
        ]
    )

    sample_ground_truth = (
        ground_truth[
            sample_index
        ]
    )


    future_frames = np.arange(

        1,

        PREDICTION_HORIZON + 1
    )


    # ========================================================
    # 4. X COORDINATE
    # ========================================================

    print()
    print("=" * 80)
    print("4. ACTUAL VS PREDICTED X")
    print("=" * 80)


    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(

        future_frames,

        sample_ground_truth[
            :,
            0
        ],

        marker="o",

        label="Actual"
    )


    plt.plot(

        future_frames,

        sample_prediction[
            :,
            0
        ],

        marker="x",

        linestyle="--",

        label="Predicted"
    )


    plt.xlabel(
        "Future Frame"
    )

    plt.ylabel(
        "X Coordinate"
    )

    plt.title(
        "Actual vs Predicted X Coordinate"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )


    save_and_show(
        "research_actual_vs_predicted_x.png"
    )


    # ========================================================
    # 5. Y COORDINATE
    # ========================================================

    print()
    print(
        "5. ACTUAL VS PREDICTED Y"
    )


    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(

        future_frames,

        sample_ground_truth[
            :,
            1
        ],

        marker="o",

        label="Actual"
    )


    plt.plot(

        future_frames,

        sample_prediction[
            :,
            1
        ],

        marker="x",

        linestyle="--",

        label="Predicted"
    )


    plt.xlabel(
        "Future Frame"
    )

    plt.ylabel(
        "Y Coordinate"
    )

    plt.title(
        "Actual vs Predicted Y Coordinate"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )


    save_and_show(
        "research_actual_vs_predicted_y.png"
    )


    # ========================================================
    # 6. Z COORDINATE
    # ========================================================

    print()
    print(
        "6. ACTUAL VS PREDICTED Z"
    )


    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(

        future_frames,

        sample_ground_truth[
            :,
            2
        ],

        marker="o",

        label="Actual"
    )


    plt.plot(

        future_frames,

        sample_prediction[
            :,
            2
        ],

        marker="x",

        linestyle="--",

        label="Predicted"
    )


    plt.xlabel(
        "Future Frame"
    )

    plt.ylabel(
        "Z Coordinate"
    )

    plt.title(
        "Actual vs Predicted Z Coordinate"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )


    save_and_show(
        "research_actual_vs_predicted_z.png"
    )


    # ========================================================
    # 7. 3D TRAJECTORY
    # ========================================================

    print()
    print(
        "7. 3D TRAJECTORY"
    )


    fig = plt.figure(
        figsize=(10, 8)
    )


    ax = fig.add_subplot(
        111,
        projection="3d"
    )


    ax.plot(

        sample_ground_truth[
            :,
            0
        ],

        sample_ground_truth[
            :,
            1
        ],

        sample_ground_truth[
            :,
            2
        ],

        marker="o",

        label="Actual"
    )


    ax.plot(

        sample_prediction[
            :,
            0
        ],

        sample_prediction[
            :,
            1
        ],

        sample_prediction[
            :,
            2
        ],

        marker="x",

        linestyle="--",

        label="Predicted"
    )


    ax.set_xlabel(
        "X"
    )

    ax.set_ylabel(
        "Y"
    )

    ax.set_zlabel(
        "Z"
    )

    ax.set_title(
        "Actual vs Predicted 3D Joint Trajectory"
    )

    ax.legend()


    save_and_show(
        "research_actual_vs_predicted_3d.png"
    )


# ============================================================
# 8. JOINT-WISE MPJPE
# ============================================================

print()
print("=" * 80)
print("8. JOINT-WISE MPJPE")
print("=" * 80)


joint_metrics_path = os.path.join(

    METRICS_DIR,

    "joint_mpjpe.csv"
)


if os.path.exists(
    joint_metrics_path
):

    joint_df = pd.read_csv(
        joint_metrics_path
    )


    plt.figure(
        figsize=(12, 6)
    )


    plt.bar(

        joint_df[
            "Joint"
        ],

        joint_df[
            "MPJPE"
        ]
    )


    plt.xlabel(
        "Joint"
    )

    plt.ylabel(
        "MPJPE"
    )

    plt.title(
        "Joint-wise Mean Per Joint Position Error"
    )

    plt.xticks(
        rotation=45
    )

    plt.grid(

        axis="y",

        alpha=0.3
    )


    save_and_show(
        "research_joint_wise_mpjpe.png"
    )

else:

    print(
        "Joint-wise metrics not found."
    )


# ============================================================
# 9–12. LSTM VS BASELINE
# ============================================================

print()
print("=" * 80)
print("LSTM VS BASELINE")
print("=" * 80)


comparison_path = os.path.join(

    METRICS_DIR,

    "comparison_metrics.csv"
)


if os.path.exists(
    comparison_path
):

    comparison_df = pd.read_csv(
        comparison_path
    )


    metrics = [

        "MAE",

        "RMSE",

        "MPJPE",

        "ADE",

        "FDE"
    ]


    for metric in metrics:

        if metric not in comparison_df[
            "Metric"
        ].values:

            continue


        row = comparison_df[
            comparison_df[
                "Metric"
            ] == metric
        ].iloc[0]


        lstm_value = row[
            "LSTM"
        ]

        baseline_value = row[
            "Last_Observed_Pose"
        ]


        plt.figure(
            figsize=(8, 6)
        )


        plt.bar(

            [
                "LSTM",
                "Last Observed Pose"
            ],

            [
                lstm_value,
                baseline_value
            ]
        )


        plt.ylabel(
            metric
        )

        plt.title(
            f"LSTM vs Last Observed Pose: "
            f"{metric}"
        )

        plt.grid(

            axis="y",

            alpha=0.3
        )


        filename = (

            "research_comparison_"

            + metric.lower()

            + ".png"
        )


        save_and_show(
            filename
        )


else:

    print(
        "Model comparison file not found."
    )


# ============================================================
# 13–15. ABLATION STUDY
# ============================================================

print()
print("=" * 80)
print("ABLATION STUDY")
print("=" * 80)


ablation_path = os.path.join(

    ABLATION_DIR,

    "ablation_results.csv"
)


if os.path.exists(
    ablation_path
):

    ablation_df = pd.read_csv(
        ablation_path
    )


    ablation_metrics = [

        "MPJPE",

        "ADE",

        "FDE"
    ]


    for metric in ablation_metrics:

        if metric not in ablation_df.columns:

            continue


        plt.figure(
            figsize=(12, 7)
        )


        plt.bar(

            ablation_df[
                "Experiment"
            ],

            ablation_df[
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
            f"Ablation Study: "
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


        save_and_show(

            f"research_ablation_"
            f"{metric.lower()}.png"
        )


else:

    print(
        "Ablation results not found."
    )


# ============================================================
# 16. ABLATION ALL METRICS
# ============================================================

if os.path.exists(
    ablation_path
):

    metrics_to_plot = [

        "MAE",

        "RMSE",

        "MPJPE",

        "ADE",

        "FDE"
    ]


    available_metrics = [

        metric

        for metric in metrics_to_plot

        if metric in ablation_df.columns
    ]


    if available_metrics:

        x = np.arange(

            len(
                ablation_df
            )
        )


        width = (

            0.8
            /
            len(
                available_metrics
            )
        )


        plt.figure(
            figsize=(14, 8)
        )


        for i, metric in enumerate(
            available_metrics
        ):

            plt.bar(

                x
                + (
                    i
                    - (
                        len(
                            available_metrics
                        )
                        - 1
                    )
                    / 2
                )
                * width,

                ablation_df[
                    metric
                ],

                width,

                label=metric
            )


        plt.xticks(

            x,

            ablation_df[
                "Experiment"
            ],

            rotation=30,

            ha="right"
        )


        plt.ylabel(
            "Error"
        )

        plt.title(
            "Feature Ablation Study"
        )

        plt.legend()

        plt.grid(

            axis="y",

            alpha=0.3
        )


        save_and_show(
            "research_ablation_all_metrics.png"
        )


# ============================================================
# FINAL REPORT TABLE
# ============================================================

print()
print("=" * 80)
print("VISUALIZATION SUMMARY")
print("=" * 80)


plot_files = [

    file

    for file in os.listdir(
        PLOTS_DIR
    )

    if file.endswith(
        ".png"
    )
]


print()

print(
    f"Total plots generated: "
    f"{len(plot_files)}"
)


for file in sorted(
    plot_files
):

    print(
        f"  - {file}"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("VISUALIZATION COMPLETED")
print("=" * 80)

print()

print(
    "All research figures are saved in:"
)

print(
    PLOTS_DIR
)

print()
print("=" * 80)