import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# SETTINGS
# ============================================================

INPUT_CSV = "data/processed/features/pose_features.csv"

OUTPUT_CSV = (
    "data/processed/features/normalized_pose_features.csv"
)

SCALER_PATH = (
    "data/processed/features/standard_scaler.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FEATURE NORMALIZATION")
print("=" * 70)

print(f"Input file: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)


# ============================================================
# BASIC CHECK
# ============================================================

print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# First two columns are metadata:
#
# 0 -> Frame Id
# 1 -> Timestamp
#
# Remaining columns are pose features.

metadata_columns = [
    "Frame Id",
    "Timestamp"
]

feature_columns = [
    column
    for column in df.columns
    if column not in metadata_columns
]


print(
    f"Feature columns: {len(feature_columns)}"
)


# ============================================================
# CHECK FEATURE COUNT
# ============================================================

if len(feature_columns) != 94:

    raise ValueError(
        f"Expected 94 feature columns, "
        f"but found {len(feature_columns)}."
    )


# ============================================================
# CHECK FOR INFINITE VALUES
# ============================================================

infinite_count = (
    df[feature_columns]
    .isin([float("inf"), float("-inf")])
    .sum()
    .sum()
)


print(
    f"Infinite values: {infinite_count}"
)


if infinite_count > 0:

    print(
        "Replacing infinite values with NaN..."
    )

    df[feature_columns] = (
        df[feature_columns]
        .replace(
            [float("inf"), float("-inf")],
            float("nan")
        )
    )


# ============================================================
# CHECK NaN VALUES
# ============================================================

total_nan = (
    df[feature_columns]
    .isna()
    .sum()
    .sum()
)


print(
    f"Total NaN values: {total_nan}"
)


# ============================================================
# HANDLE NaN VALUES
# ============================================================

"""
For the first normalization stage we use
forward-fill followed by backward-fill.

Why?

Pose estimation can occasionally fail for a frame.
We don't want isolated missing values to remain in
the normalized dataset.

However, we do NOT fill everything blindly with zero.
"""

df[feature_columns] = (
    df[feature_columns]
    .ffill()
    .bfill()
)


# ============================================================
# FINAL NaN CHECK
# ============================================================

remaining_nan = (
    df[feature_columns]
    .isna()
    .sum()
    .sum()
)


print(
    f"Remaining NaN values: {remaining_nan}"
)


if remaining_nan > 0:

    raise ValueError(
        "NaN values still remain after preprocessing."
    )


# ============================================================
# CREATE SCALER
# ============================================================

scaler = StandardScaler()


# ============================================================
# FIT + TRANSFORM
# ============================================================

scaled_features = scaler.fit_transform(
    df[feature_columns]
)


# ============================================================
# CREATE NORMALIZED DATAFRAME
# ============================================================

normalized_features = pd.DataFrame(
    scaled_features,
    columns=feature_columns,
    index=df.index
)


# ============================================================
# COMBINE METADATA + NORMALIZED FEATURES
# ============================================================

normalized_df = pd.concat(
    [
        df[metadata_columns],
        normalized_features
    ],
    axis=1
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_CSV),
    exist_ok=True
)


# ============================================================
# SAVE NORMALIZED CSV
# ============================================================

normalized_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    SCALER_PATH
)


# ============================================================
# VERIFICATION
# ============================================================

print()
print("=" * 70)
print("NORMALIZATION COMPLETED")
print("=" * 70)

print(
    f"Rows                  : "
    f"{len(normalized_df)}"
)

print(
    f"Features normalized   : "
    f"{len(feature_columns)}"
)

print(
    f"Normalized CSV        : "
    f"{OUTPUT_CSV}"
)

print(
    f"Scaler saved          : "
    f"{SCALER_PATH}"
)

print("=" * 70)