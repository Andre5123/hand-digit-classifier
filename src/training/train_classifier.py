import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Turns off oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Turns off all TensorFlow C++ logs (0 = all, 1 = info, 2 = warnings, 3 = errors only)
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # To be able to access files in another folder

import tensorflow as tf

from models.classifier import build_classifier
from data.preprocess import make_custom_classification_datasets
import math

#model = build_classifier() #Toggle to restart a new model
model = tf.keras.models.load_model("../../models/classifier_best.keras") # Toggle to keep improving on existing model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), #Start with 0.001 initially.
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy']
)

train_dataset, val_dataset, test_dataset, train_size, val_size, test_size = make_custom_classification_datasets("../../data/custom/crops", 32, 0.7)

model.summary()


print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")



model.fit(
    train_dataset,
    epochs=100,
    steps_per_epoch=math.ceil(train_size/32),
    validation_data=val_dataset,
    validation_steps=math.ceil(val_size/32),
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            filepath='../../models/classifier_best.keras',
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=15,
            monitor='val_accuracy',
            mode='max',
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=8,
            monitor='val_accuracy',
            mode='max',
            min_lr=1e-5
        )
    ]
)



