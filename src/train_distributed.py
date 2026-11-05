"""
Real-time use case:
Factory defect detection model retrained daily from newly collected camera images.

This script demonstrates:
- Keras + tf.data best practices
- Distributed training with tf.distribute.MirroredStrategy
- Production-friendly callbacks and model export
"""

import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------
# Config
# ----------------------------
DATA_DIR = os.getenv("DATA_DIR", "./data/images")  # expected structure: class folders
IMG_SIZE = (224, 224)
BATCH_SIZE_PER_REPLICA = 32
EPOCHS = int(os.getenv("EPOCHS", "5"))
MODEL_DIR = os.getenv("MODEL_DIR", "./artifacts")
os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------------------
# Distributed strategy
# ----------------------------
strategy = tf.distribute.MirroredStrategy()
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

print(f"Replicas in sync: {strategy.num_replicas_in_sync}")
print(f"Global batch size: {GLOBAL_BATCH_SIZE}")

# ----------------------------
# Data pipeline (tf.data best practices)
# ----------------------------
train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=GLOBAL_BATCH_SIZE,
)

val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=GLOBAL_BATCH_SIZE,
)

class_names = train_ds.class_names
num_classes = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE

def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    return x, y

train_ds = (
    train_ds
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .shuffle(1000)
    .prefetch(AUTOTUNE)
)

val_ds = (
    val_ds
    .map(preprocess, num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

# ----------------------------
# Model (inside strategy scope)
# ----------------------------
with strategy.scope():
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ])

    base = keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

callbacks = [
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.2),
    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "best_model.keras"),
        save_best_only=True,
        monitor="val_accuracy"
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Export final model and training summary
final_model_path = os.path.join(MODEL_DIR, "final_model.keras")
model.save(final_model_path)

report = {
    "class_names": class_names,
    "epochs_ran": len(history.history["loss"]),
    "final_val_accuracy": float(history.history["val_accuracy"][-1]),
    "replicas_in_sync": strategy.num_replicas_in_sync,
    "global_batch_size": GLOBAL_BATCH_SIZE,
    "model_path": final_model_path
}

with open(os.path.join(MODEL_DIR, "training_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("Training complete.")
print(json.dumps(report, indent=2))