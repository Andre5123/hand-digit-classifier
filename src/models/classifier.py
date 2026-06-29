import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def conv_block(x, n_filters):
    x = layers.Conv2D(filters=n_filters, kernel_size=(3,3), strides=(1,1), padding="same", activation=None)(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=n_filters, kernel_size=(3,3), strides=(1,1), padding="same", activation=None)(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.ReLU()(x)
    output = layers.MaxPool2D(pool_size=(2,2))(x)
    return output

def build_classifier(input_shape=(128,128,1), num_classes=6, dropout_rate=0.5):
    
    image_inputs = keras.Input(shape=input_shape)
    x = conv_block(image_inputs, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    x = layers.Flatten()(x)
    x = layers.Dense(units=512, activation='relu')(x)
    x = layers.Dropout(rate=dropout_rate)(x)
    outputs = layers.Dense(units=num_classes, activation='softmax')(x)

    model = keras.Model(inputs=image_inputs, outputs=outputs, name="classifier_model")
    return model
