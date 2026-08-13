import os

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASELINE_PATH = (
    "data/processed/evaluation/metrics/"
    "baseline_results.csv"
)

LSTM_OVERALL_PATH = (
    "data/processed/evaluation/metrics/"
    "overall_metrics.csv"
)

LSTM_POSE_PATH = (
    "data/processed/evaluation/metrics/"
    "pose_metrics.csv"
)

OUTPUT_DIR = (
    "data/processed/evaluation/metrics"
)

PLOT_DIR = (
    "data/processed/evaluation/plots"
)


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


# ============================================================
# LOAD BASELINE
# ============================================================

print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print()
print("Loading baseline results...")

baseline_df = pd.read_csv(
    BASELINE_PATH
)

print(
    baseline_df.to_string(
        index=False
    )
)


# ============================================================
# LOAD LSTM OVERALL METRICS
# ============================================================

print()
print("Loading LSTM metrics...")

lstm_overall_df = pd.read_csv(
    LSTM_OVERALL_PATH
)

lstm_pose_df = pd.read_csv(
    LSTM_POSE_PATH
)


# ============================================================
# EXTRACT LSTM METRICS
# ============================================================

lstm_metrics = {}

for _, row in lstm_overall_df.iterrows():

    lstm_metrics[
        row["Metric"]
    ] = row["Value"]


for _, row in lstm_pose_df.iterrows():

    lstm_metrics[
        row["Metric"]
    ] = row["Value"]


# ============================================================
# EXTRACT BASELINE METRICS
# ============================================================

baseline_row = baseline_df.iloc[0]


baseline_metrics = {

    "MAE":
        baseline_row["MAE"],

    "MSE":
        baseline_row["MSE"],

    "RMSE":
        baseline_row["RMSE"],

    "MPJPE":
        baseline_row["MPJPE"],

    "ADE":
        baseline_row["ADE"],

    "FDE":
        baseline_row["FDE"]
}


# ============================================================
# CREATE COMPARISON TABLE
# ============================================================

metrics = [

    "MAE",
    "MSE",
    "RMSE",
    "MPJPE",
    "ADE",
    "FDE"
]


comparison_rows = []


for metric in metrics:

    comparison_rows.append({

        "Metric":
            metric,

        "Last_Pose_Baseline":
            baseline_metrics[metric],

        "LSTM":
            lstm_metrics[metric]
    })


comparison_df = pd.DataFrame(
    comparison_rows
)


# ============================================================
# CALCULATE IMPROVEMENT
# ============================================================

"""
Percentage improvement:

    ((Baseline - LSTM) / Baseline) * 100

Positive value:
    LSTM is better.

Negative value:
    Baseline is better.
"""


comparison_df[
    "Improvement_Percent"
] = (

    (
        comparison_df[
            "Last_Pose_Baseline"
        ]

        -

        comparison_df[
            "LSTM"
        ]
    )

    /

    comparison_df[
        "Last_Pose_Baseline"
    ]

) * 100


# ============================================================
# PRINT COMPARISON
# ============================================================

print()
print("=" * 70)
print("BASELINE VS LSTM")
print("=" * 70)

print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE CSV
# ============================================================

comparison_path = os.path.join(

    OUTPUT_DIR,

    "comparison_results.csv"
)


comparison_df.to_csv(

    comparison_path,

    index=False
)


print()
print(
    "Comparison saved:"
)

print(
    comparison_path
)


# ============================================================
# CREATE COMPARISON GRAPH
# ============================================================

plot_df = comparison_df[
    comparison_df["Metric"].isin(
        [
            "MAE",
            "RMSE",
            "MPJPE",
            "ADE",
            "FDE"
        ]
    )
]


plt.figure(
    figsize=(12, 6)
)


x = range(
    len(plot_df)
)


width = 0.35


plt.bar(

    [
        i - width / 2
        for i in x
    ],

    plot_df[
        "Last_Pose_Baseline"
    ],

    width=width,

    label="Last Pose Baseline"
)


plt.bar(

    [
        i + width / 2
        for i in x
    ],

    plot_df[
        "LSTM"
    ],

    width=width,

    label="LSTM"
)


plt.xticks(

    list(x),

    plot_df[
        "Metric"
    ]
)


plt.xlabel(
    "Evaluation Metric"
)

plt.ylabel(
    "Error"
)

plt.title(
    "Last Pose Baseline vs LSTM"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


comparison_plot = os.path.join(

    PLOT_DIR,

    "baseline_vs_lstm.png"
)


plt.savefig(

    comparison_plot,

    dpi=200,

    bbox_inches="tight"
)


print(
    "Comparison graph saved:"
)

print(
    comparison_plot
)


plt.show()

plt.close()


# ============================================================
# IMPROVEMENT GRAPH
# ============================================================

plt.figure(
    figsize=(11, 6)
)


plt.bar(

    comparison_df[
        "Metric"
    ],

    comparison_df[
        "Improvement_Percent"
    ]
)


plt.axhline(
    0,
    linewidth=1
)


plt.xlabel(
    "Metric"
)

plt.ylabel(
    "Improvement (%)"
)

plt.title(
    "LSTM Improvement over Last-Pose Baseline"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


improvement_plot = os.path.join(

    PLOT_DIR,

    "lstm_improvement_over_baseline.png"
)


plt.savefig(

    improvement_plot,

    dpi=200,

    bbox_inches="tight"
)


print(
    "Improvement graph saved:"
)

print(
    improvement_plot
)


plt.show()

plt.close()


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("MODEL COMPARISON COMPLETED")
print("=" * 70)

print()
print(
    "Files:"
)

print(
    comparison_path
)

print(
    comparison_plot
)

print(
    improvement_plot
)

print("=" * 70)