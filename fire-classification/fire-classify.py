#!/usr/bin/env python3
import os, json, numpy as np, tensorflow as tf
from tensorflow import keras
from pathlib import Path

# Optional: Force CPU if GPU/MPS causes memory spikes
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

#github data folder is empty due to storage limitations; can be found on Kaggle
TRAIN_DIR = "fire-classification/data/train"
TEST_DIR  = "fire-classification/data/test"
IMG_SIZE  = 160        # smaller image size
BATCH     = 8          # smaller batch size
EPOCHS    = 8          # fewer epochs initially
MODEL_PATH  = "fire-classification/fire-classify.keras"
LABELS_PATH = "fire-classification/labels.json"

# 1) Train/Val loaders (10% val from train)
train_full = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    label_mode="categorical",
    validation_split=0.10,
    subset="training",
    seed=42
)
val_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    label_mode="categorical",
    validation_split=0.10,
    subset="validation",
    seed=42
)

# 2) Test loader (independent)
test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    label_mode="categorical",
    shuffle=False
)

class_names = train_full.class_names
print("Classes (train):", class_names)
print("Classes (test): ", test_ds.class_names)
assert class_names == test_ds.class_names, "Train/test class sets must match."

with open(LABELS_PATH, "w") as f:
    json.dump(class_names, f)

# Light prefetch only — no caching to reduce memory usage
train_ds = train_full.shuffle(256).prefetch(1)
val_ds   = val_ds.prefetch(1)
test_ds  = test_ds.prefetch(1)

# 3) Model (MobileNetV2 smaller version)
base = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
    alpha=0.5  # half-width MobileNet for smaller memory footprint
)
base.trainable = False

aug = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.05),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomContrast(0.1),
])

inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = aug(inp)
x = keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
out = keras.layers.Dense(len(class_names), activation="softmax")(x)
model = keras.Model(inp, out)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    run_eagerly=False  # keeps memory lower
)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# 4) Light fine-tuning (optional; small unfreeze)
base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(train_ds, validation_data=val_ds, epochs=2)

# 5) Save model + labels
model.save(MODEL_PATH)
print("Saved model:", MODEL_PATH)
print("Saved labels:", LABELS_PATH)

# 6) Evaluate on TEST
y_true = []
for _, y in test_ds:
    y_true.append(y.numpy())
y_true = np.argmax(np.concatenate(y_true, axis=0), axis=1)

probs = model.predict(test_ds, verbose=0)
y_pred = np.argmax(probs, axis=1)

acc = (y_pred == y_true).mean()
print(f"TEST accuracy: {acc*100:.2f}%")

cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=len(class_names)).numpy()
print("Confusion matrix (rows=true, cols=pred):")
print(cm)
print("Labels order:", class_names)
