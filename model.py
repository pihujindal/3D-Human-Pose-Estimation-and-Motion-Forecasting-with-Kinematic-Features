import os

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    RepeatVector,
    TimeDistributed
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_DIR = (
    "data/processed/sequences"
)

MODEL_DIR = (
    "models"
)

MODEL_PATH = (
    "models/lstm_motion_forecasting.keras"
)

HISTORY_PATH = (
    "models/training_history.npy"
)


# ============================================================
# DATA PARAMETERS
# ============================================================

INPUT_WINDOW = 30

PREDICTION_HORIZON = 10

N_FEATURES = 94


# ============================================================
# TRAINING PARAMETERS
# ============================================================

EPOCHS = 100

BATCH_SIZE = 32

LEARNING_RATE = 0.001


# ============================================================
# RANDOM SEED
# ============================================================

"""
Set seeds so that experiments are more reproducible.
"""

SEED = 42

np.random.seed(SEED)

tf.random.set_seed(SEED)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("LSTM HUMAN MOTION FORECASTING")
print("=" * 70)

print()
print("Loading prepared sequences...")


X_train = np.load(
    os.path.join(
        SEQUENCE_DIR,
        "X_train.npy"
    )
)

y_train = np.load(
    os.path.join(
        SEQUENCE_DIR,
        "y_train.npy"
    )
)


X_val = np.load(
    os.path.join(
        SEQUENCE_DIR,
        "X_val.npy"
    )
)

y_val = np.load(
    os.path.join(
        SEQUENCE_DIR,
        "y_val.npy"
    )
)


X_test = np.load(
    os.path.join(
        SEQUENCE_DIR,
        "X_test.npy"
    )
)

y_test = np.load(
    os.path.join(
        SEQUENCE_DIR,
        "y_test.npy"
    )
)


# ============================================================
# DISPLAY SHAPES
# ============================================================

print()
print("=" * 70)
print("DATASET SHAPES")
print("=" * 70)

print(
    f"X_train : {X_train.shape}"
)

print(
    f"y_train : {y_train.shape}"
)

print(
    f"X_val   : {X_val.shape}"
)

print(
    f"y_val   : {y_val.shape}"
)

print(
    f"X_test  : {X_test.shape}"
)

print(
    f"y_test  : {y_test.shape}"
)


# ============================================================
# VALIDATE DATA DIMENSIONS
# ============================================================

datasets = {

    "X_train": X_train,

    "y_train": y_train,

    "X_val": X_val,

    "y_val": y_val,

    "X_test": X_test,

    "y_test": y_test
}


for name, array in datasets.items():

    if array.ndim != 3:

        raise ValueError(

            f"{name} must be 3-dimensional.\n"

            f"Received shape: {array.shape}"
        )


# ============================================================
# VALIDATE INPUT SHAPES
# ============================================================

if X_train.shape[1] != INPUT_WINDOW:

    raise ValueError(

        f"X_train must contain "
        f"{INPUT_WINDOW} frames.\n"

        f"Received: "
        f"{X_train.shape[1]}"
    )


if X_train.shape[2] != N_FEATURES:

    raise ValueError(

        f"X_train must contain "
        f"{N_FEATURES} features.\n"

        f"Received: "
        f"{X_train.shape[2]}"
    )


# ============================================================
# VALIDATE TARGET SHAPES
# ============================================================

if y_train.shape[1] != PREDICTION_HORIZON:

    raise ValueError(

        f"y_train must contain "
        f"{PREDICTION_HORIZON} future frames.\n"

        f"Received: "
        f"{y_train.shape[1]}"
    )


if y_train.shape[2] != N_FEATURES:

    raise ValueError(

        f"y_train must contain "
        f"{N_FEATURES} features.\n"

        f"Received: "
        f"{y_train.shape[2]}"
    )


# ============================================================
# VALIDATE VALIDATION DATA
# ============================================================

if X_val.shape[1:] != (
    INPUT_WINDOW,
    N_FEATURES
):

    raise ValueError(
        f"Invalid X_val shape: "
        f"{X_val.shape}"
    )


if y_val.shape[1:] != (
    PREDICTION_HORIZON,
    N_FEATURES
):

    raise ValueError(
        f"Invalid y_val shape: "
        f"{y_val.shape}"
    )


# ============================================================
# VALIDATE TEST DATA
# ============================================================

if X_test.shape[1:] != (
    INPUT_WINDOW,
    N_FEATURES
):

    raise ValueError(
        f"Invalid X_test shape: "
        f"{X_test.shape}"
    )


if y_test.shape[1:] != (
    PREDICTION_HORIZON,
    N_FEATURES
):

    raise ValueError(
        f"Invalid y_test shape: "
        f"{y_test.shape}"
    )


# ============================================================
# CHECK NaN VALUES
# ============================================================

print()
print(
    "Checking for NaN values..."
)


for name, array in datasets.items():

    if np.isnan(array).any():

        raise ValueError(

            f"NaN values found in "
            f"{name}."
        )


print(
    "No NaN values found."
)


# ============================================================
# CHECK INFINITE VALUES
# ============================================================

print(
    "Checking for infinite values..."
)


for name, array in datasets.items():

    if not np.isfinite(array).all():

        raise ValueError(

            f"Infinite values found "
            f"in {name}."
        )


print(
    "No infinite values found."
)


# ============================================================
# BUILD ENCODER-DECODER LSTM
# ============================================================

print()
print("=" * 70)
print("BUILDING LSTM MODEL")
print("=" * 70)


model = Sequential()


# ============================================================
# ENCODER - LSTM 1
# ============================================================

model.add(

    LSTM(

        128,

        return_sequences=True,

        input_shape=(

            INPUT_WINDOW,

            N_FEATURES
        )
    )
)


model.add(
    Dropout(0.2)
)


# ============================================================
# ENCODER - LSTM 2
# ============================================================

model.add(

    LSTM(

        128,

        return_sequences=False
    )
)


model.add(
    Dropout(0.2)
)


# ============================================================
# REPEAT ENCODED REPRESENTATION
# ============================================================

model.add(

    RepeatVector(

        PREDICTION_HORIZON
    )
)


# ============================================================
# DECODER
# ============================================================

model.add(

    LSTM(

        128,

        return_sequences=True
    )
)


model.add(
    Dropout(0.2)
)


# ============================================================
# OUTPUT LAYER
# ============================================================

model.add(

    TimeDistributed(

        Dense(
            N_FEATURES
        )
    )
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = tf.keras.optimizers.Adam(

    learning_rate=LEARNING_RATE
)


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=optimizer,

    loss="mse",

    metrics=[
        "mae"
    ]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

print()
print("=" * 70)
print("MODEL ARCHITECTURE")
print("=" * 70)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=15,

    restore_best_weights=True,

    verbose=1
)


# ------------------------------------------------------------
# Save best model
# ------------------------------------------------------------

checkpoint = ModelCheckpoint(

    MODEL_PATH,

    monitor="val_loss",

    save_best_only=True,

    verbose=1
)


# ------------------------------------------------------------
# Reduce learning rate when validation loss stops improving
# ------------------------------------------------------------

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=5,

    min_lr=1e-6,

    verbose=1
)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("TRAINING")
print("=" * 70)

print()
print(
    f"Epochs     : {EPOCHS}"
)

print(
    f"Batch size : {BATCH_SIZE}"
)

print(
    f"Learning rate : {LEARNING_RATE}"
)


history = model.fit(

    X_train,

    y_train,

    validation_data=(

        X_val,

        y_val
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[

        early_stopping,

        checkpoint,

        reduce_lr
    ],

    verbose=1
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)


best_model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# TEST EVALUATION
# ============================================================

print()
print("=" * 70)
print("TEST EVALUATION")
print("=" * 70)


test_loss, test_mae = (
    best_model.evaluate(

        X_test,

        y_test,

        verbose=1
    )
)


print()
print(
    f"Test MSE : {test_loss:.6f}"
)

print(
    f"Test MAE : {test_mae:.6f}"
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

print()
print(
    "Saving training history..."
)


np.save(

    HISTORY_PATH,

    history.history,

    allow_pickle=True
)


print(
    "Training history saved:"
)

print(
    HISTORY_PATH
)


# ============================================================
# SAVE MODEL INFORMATION
# ============================================================

model_info = {

    "input_window":
        INPUT_WINDOW,

    "prediction_horizon":
        PREDICTION_HORIZON,

    "n_features":
        N_FEATURES,

    "epochs":
        EPOCHS,

    "batch_size":
        BATCH_SIZE,

    "learning_rate":
        LEARNING_RATE,

    "random_seed":
        SEED,

    "best_validation_loss":
        float(
            min(
                history.history[
                    "val_loss"
                ]
            )
        ),

    "test_mse":
        float(
            test_loss
        ),

    "test_mae":
        float(
            test_mae
        )
}


np.save(

    os.path.join(
        MODEL_DIR,
        "model_info.npy"
    ),

    model_info,

    allow_pickle=True
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("LSTM TRAINING COMPLETED")
print("=" * 70)

print()
print(
    "Best model:"
)

print(
    MODEL_PATH
)

print()
print(
    "Training history:"
)

print(
    HISTORY_PATH
)

print()
print(
    "Model information:"
)

print(
    os.path.join(
        MODEL_DIR,
        "model_info.npy"
    )
)

print()
print(
    "The detailed research evaluation should now be "
    "performed using:"
)

print(
    "python src\\evaluation.py"
)

print("=" * 70)