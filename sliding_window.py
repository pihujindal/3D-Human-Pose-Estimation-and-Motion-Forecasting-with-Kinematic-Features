import os

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Input file
# ------------------------------------------------------------

INPUT_CSV = (
    "data/processed/features/"
    "normalized_pose_data.csv"
)


# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

OUTPUT_DIR = (
    "data/processed/sequences"
)


# ------------------------------------------------------------
# Temporal parameters
# ------------------------------------------------------------

INPUT_WINDOW = 30

PREDICTION_HORIZON = 10


# ------------------------------------------------------------
# Dataset split
# ------------------------------------------------------------

TRAIN_RATIO = 0.70

VALIDATION_RATIO = 0.15

TEST_RATIO = 0.15


# ------------------------------------------------------------
# Number of features
# ------------------------------------------------------------

N_FEATURES = 94


# ============================================================
# VALIDATE SPLIT RATIOS
# ============================================================

total_ratio = (

    TRAIN_RATIO
    + VALIDATION_RATIO
    + TEST_RATIO
)


if not np.isclose(
    total_ratio,
    1.0
):

    raise ValueError(
        "TRAIN_RATIO + VALIDATION_RATIO "
        "+ TEST_RATIO must equal 1.0"
    )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FIND INPUT FILE
# ============================================================

if not os.path.exists(
    INPUT_CSV
):

    alternative_paths = [

        "normalized_pose_data.csv",

        "data/processed/"
        "normalized_pose_data.csv",

        "data/processed/features/"
        "normalized_pose_features.csv"
    ]


    found_path = None


    for path in alternative_paths:

        if os.path.exists(path):

            found_path = path

            break


    if found_path is None:

        raise FileNotFoundError(

            "\nCould not find normalized "
            "pose CSV.\n\n"

            "Expected:\n"
            f"{INPUT_CSV}\n\n"

            "Also checked:\n"
            + "\n".join(
                alternative_paths
            )
        )


    INPUT_CSV = found_path


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("SLIDING WINDOW GENERATION")
print("=" * 70)

print()
print("Input file:")
print(INPUT_CSV)

print()
print("Loading normalized feature data...")


df = pd.read_csv(
    INPUT_CSV
)


print(
    f"Rows: {len(df)}"
)

print(
    f"Columns: {len(df.columns)}"
)


# ============================================================
# DISPLAY COLUMNS
# ============================================================

print()
print("Columns:")

for index, column in enumerate(
    df.columns
):

    print(
        f"{index:3d} : {column}"
    )


# ============================================================
# IDENTIFY FEATURE COLUMNS
# ============================================================

"""
Normally the first two columns are:

    Frame Id
    Timestamp

Everything else is treated as a feature.

However, this code does not blindly assume
that exactly two metadata columns exist.

Known metadata columns are removed.
"""

metadata_candidates = [

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

    for column in df.columns

    if column not in metadata_candidates
]


# ============================================================
# CHECK FEATURE COUNT
# ============================================================

print()
print(
    "Detected feature count:",
    len(feature_columns)
)


if len(feature_columns) != N_FEATURES:

    raise ValueError(

        "\nExpected exactly "
        f"{N_FEATURES} features.\n"

        f"Detected: "
        f"{len(feature_columns)}\n\n"

        "Detected feature columns:\n"

        + "\n".join(
            feature_columns
        )
    )


# ============================================================
# CHECK NUMERIC FEATURES
# ============================================================

print()
print(
    "Checking feature data..."
)


for column in feature_columns:

    if not pd.api.types.is_numeric_dtype(
        df[column]
    ):

        raise TypeError(

            f"Feature '{column}' "
            "is not numeric."
        )


# ============================================================
# CHECK MISSING VALUES
# ============================================================

missing_values = (
    df[feature_columns]
    .isna()
    .sum()
    .sum()
)


print(
    "Total missing feature values:",
    missing_values
)


if missing_values > 0:

    print()
    print(
        "WARNING:"
    )

    print(
        "Missing values detected."
    )

    print(
        "Filling them using forward-fill "
        "followed by backward-fill."
    )


    df[feature_columns] = (
        df[feature_columns]
        .ffill()
        .bfill()
    )


# ============================================================
# EXTRACT FEATURE MATRIX
# ============================================================

features = (
    df[
        feature_columns
    ]
    .to_numpy(
        dtype=np.float32
    )
)


print()
print(
    "Feature matrix shape:",
    features.shape
)


# ============================================================
# CHECK ENOUGH FRAMES
# ============================================================

minimum_frames = (

    INPUT_WINDOW
    + PREDICTION_HORIZON
)


if len(features) < minimum_frames:

    raise ValueError(

        f"Not enough frames.\n"

        f"Required at least: "
        f"{minimum_frames}\n"

        f"Available: "
        f"{len(features)}"
    )


# ============================================================
# TOTAL POSSIBLE WINDOWS
# ============================================================

total_windows = (

    len(features)
    - INPUT_WINDOW
    - PREDICTION_HORIZON
    + 1
)


print()
print(
    "Total possible sliding windows:",
    total_windows
)


# ============================================================
# CHRONOLOGICAL DATA SPLIT
# ============================================================

"""
We split the ORIGINAL FRAME SEQUENCE first.

This is important.

We do NOT create all overlapping windows and then
randomly split them.

Why?

Because overlapping windows would leak almost identical
frames between training and testing.

Example of bad split:

Training:
    frames 1-30

Testing:
    frames 2-31

These two samples share 29 frames.

That produces temporal leakage.

Instead:

    first 70% → training
    next 15%  → validation
    final 15% → testing

Then sliding windows are generated independently
inside each split.
"""


total_frames = len(features)


train_end = int(
    total_frames * TRAIN_RATIO
)


validation_end = (

    train_end
    + int(
        total_frames
        * VALIDATION_RATIO
    )
)


train_data = features[
    :train_end
]


validation_data = features[
    train_end:validation_end
]


test_data = features[
    validation_end:
]


print()
print("=" * 70)
print("CHRONOLOGICAL DATA SPLIT")
print("=" * 70)

print()
print(
    f"Total frames      : {total_frames}"
)

print(
    f"Training frames   : {len(train_data)}"
)

print(
    f"Validation frames : {len(validation_data)}"
)

print(
    f"Testing frames    : {len(test_data)}"
)


# ============================================================
# CHECK SPLIT SIZES
# ============================================================

for name, data in [

    ("Training", train_data),

    ("Validation", validation_data),

    ("Testing", test_data)

]:

    if len(data) < minimum_frames:

        raise ValueError(

            f"{name} split contains only "
            f"{len(data)} frames.\n"

            f"At least {minimum_frames} "
            "frames are required to create "
            "one complete sample."
        )


# ============================================================
# SLIDING WINDOW FUNCTION
# ============================================================

def create_sliding_windows(
    data,
    input_window,
    prediction_horizon
):
    """
    Create input-target pairs.

    Input:
        30 consecutive frames

    Target:
        next 10 consecutive frames

    Example:

        data[0:30]
              ↓
        data[30:40]

        data[1:31]
              ↓
        data[31:41]

        data[2:32]
              ↓
        data[32:42]
    """

    X = []

    y = []


    total_samples = (

        len(data)
        - input_window
        - prediction_horizon
        + 1
    )


    for start in range(
        total_samples
    ):

        # ----------------------------------------------------
        # Input sequence
        # ----------------------------------------------------

        input_start = start

        input_end = (
            start
            + input_window
        )


        # ----------------------------------------------------
        # Future target sequence
        # ----------------------------------------------------

        target_start = input_end

        target_end = (

            input_end
            + prediction_horizon
        )


        # ----------------------------------------------------
        # Extract
        # ----------------------------------------------------

        input_sequence = data[
            input_start:input_end
        ]


        target_sequence = data[
            target_start:target_end
        ]


        X.append(
            input_sequence
        )

        y.append(
            target_sequence
        )


    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    X = np.asarray(
        X,
        dtype=np.float32
    )


    y = np.asarray(
        y,
        dtype=np.float32
    )


    return X, y


# ============================================================
# CREATE TRAINING WINDOWS
# ============================================================

print()
print(
    "Creating training windows..."
)


X_train, y_train = (
    create_sliding_windows(

        train_data,

        INPUT_WINDOW,

        PREDICTION_HORIZON
    )
)


print(
    "X_train:",
    X_train.shape
)

print(
    "y_train:",
    y_train.shape
)


# ============================================================
# CREATE VALIDATION WINDOWS
# ============================================================

print()
print(
    "Creating validation windows..."
)


X_val, y_val = (
    create_sliding_windows(

        validation_data,

        INPUT_WINDOW,

        PREDICTION_HORIZON
    )
)


print(
    "X_val:",
    X_val.shape
)

print(
    "y_val:",
    y_val.shape
)


# ============================================================
# CREATE TEST WINDOWS
# ============================================================

print()
print(
    "Creating test windows..."
)


X_test, y_test = (
    create_sliding_windows(

        test_data,

        INPUT_WINDOW,

        PREDICTION_HORIZON
    )
)


print(
    "X_test:",
    X_test.shape
)

print(
    "y_test:",
    y_test.shape
)


# ============================================================
# FINAL SHAPE VALIDATION
# ============================================================

expected_train_x_features = (
    X_train.shape[2]
)

expected_train_y_features = (
    y_train.shape[2]
)


if X_train.ndim != 3:

    raise ValueError(
        "X_train is not 3-dimensional."
    )


if y_train.ndim != 3:

    raise ValueError(
        "y_train is not 3-dimensional."
    )


if X_train.shape[1] != INPUT_WINDOW:

    raise ValueError(
        "Incorrect X_train input window."
    )


if y_train.shape[1] != PREDICTION_HORIZON:

    raise ValueError(
        "Incorrect y_train prediction horizon."
    )


if expected_train_x_features != N_FEATURES:

    raise ValueError(
        "Incorrect X_train feature count."
    )


if expected_train_y_features != N_FEATURES:

    raise ValueError(
        "Incorrect y_train feature count."
    )


# ============================================================
# SAVE NUMPY FILES
# ============================================================

print()
print("=" * 70)
print("SAVING SEQUENCES")
print("=" * 70)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_train.npy"
    ),
    X_train
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_train.npy"
    ),
    y_train
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_val.npy"
    ),
    X_val
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_val.npy"
    ),
    y_val
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_test.npy"
    ),
    X_test
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_test.npy"
    ),
    y_test
)


# ============================================================
# METADATA
# ============================================================

metadata = {

    "input_file":
        INPUT_CSV,

    "total_frames":
        total_frames,

    "total_features":
        N_FEATURES,

    "input_window":
        INPUT_WINDOW,

    "prediction_horizon":
        PREDICTION_HORIZON,

    "train_ratio":
        TRAIN_RATIO,

    "validation_ratio":
        VALIDATION_RATIO,

    "test_ratio":
        TEST_RATIO,

    "train_frames":
        len(train_data),

    "validation_frames":
        len(validation_data),

    "test_frames":
        len(test_data),

    "train_samples":
        len(X_train),

    "validation_samples":
        len(X_val),

    "test_samples":
        len(X_test)
}


metadata_df = pd.DataFrame(

    list(
        metadata.items()
    ),

    columns=[
        "Parameter",
        "Value"
    ]
)


metadata_path = os.path.join(

    OUTPUT_DIR,

    "sequence_metadata.csv"
)


metadata_df.to_csv(

    metadata_path,

    index=False
)


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("SEQUENCE GENERATION COMPLETED")
print("=" * 70)

print()

print(
    "Training:"
)

print(
    f"X_train = {X_train.shape}"
)

print(
    f"y_train = {y_train.shape}"
)


print()

print(
    "Validation:"
)

print(
    f"X_val   = {X_val.shape}"
)

print(
    f"y_val   = {y_val.shape}"
)


print()

print(
    "Testing:"
)

print(
    f"X_test  = {X_test.shape}"
)

print(
    f"y_test  = {y_test.shape}"
)


print()
print(
    "Feature count:",
    N_FEATURES
)

print(
    "Input window:",
    INPUT_WINDOW
)

print(
    "Prediction horizon:",
    PREDICTION_HORIZON
)


print()
print(
    "Saved files:"
)

print(
    OUTPUT_DIR
)

print(
    "  ├── X_train.npy"
)

print(
    "  ├── y_train.npy"
)

print(
    "  ├── X_val.npy"
)

print(
    "  ├── y_val.npy"
)

print(
    "  ├── X_test.npy"
)

print(
    "  ├── y_test.npy"
)

print(
    "  └── sequence_metadata.csv"
)


print()
print("=" * 70)