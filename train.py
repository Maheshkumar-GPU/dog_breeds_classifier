import os
import json
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.preprocess import load_dataset
from src.model import build_model
from src.utils import plot_results
from config import model_path, label_map_path, results_path, EPO, LEARNING_RATE

print("Loading dataset...")


train_dataset, val_dataset, class_names = load_dataset()


AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

print("Dataset loaded")

model = build_model(len(class_names))

model.summary()


checkpoint = ModelCheckpoint(
    model_path,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.3,
    patience=3,
    verbose=1
)


print("Training started...")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPO,
    callbacks=[checkpoint, early_stop, lr_scheduler]
)

print("Training completed")

label_map = {name: i for i, name in enumerate(class_names)}

with open(label_map_path, "w") as f:
    json.dump(label_map, f)

plot_results(history)

print(" All done successfully! you can predict the breed")