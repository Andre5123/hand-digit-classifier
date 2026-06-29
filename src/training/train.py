import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Turns off oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Turns off all TensorFlow C++ logs (0 = all, 1 = info, 2 = warnings, 3 = errors only)
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # To be able to access files in another folder

import tensorflow as tf

from models.classifier import build_classifier
from data.preprocess import make_classification_dataset

import math

model = build_classifier()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy']
)

train_dataset, train_size = make_classification_dataset("../../data/raw/digits/train",32)
valid_dataset, val_size  = make_classification_dataset("../../data/raw/digits/valid",32)

model.fit(
    train_dataset,
    epochs=10,
    steps_per_epoch=math.ceil(train_size/32),
    validation_data=valid_dataset,
    validation_steps=math.ceil(val_size/32),
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            filepath='../../models/classifier_best.keras',
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=5,
            monitor='val_accuracy',
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=3,
            monitor='val_accuracy'
        )
    ]
)



