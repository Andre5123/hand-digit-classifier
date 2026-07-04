import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

def conv_block(x, n_filters, l2=0.001):
    x = layers.Conv2D(filters=n_filters, kernel_size=(3,3), strides=(1,1), padding="same", activation=None, kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=n_filters, kernel_size=(3,3), strides=(1,1), padding="same", activation=None, kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.ReLU()(x)
    output = layers.MaxPool2D(pool_size=(2,2))(x)
    return output

def build_detector(input_shape=(256,256, 3)):
    image_inputs = keras.Input(shape=input_shape)
    x = conv_block(image_inputs, 16)
    x = conv_block(x, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)
    outputs = layers.Conv2D(filters=5, kernel_size =(1,1), strides=(1,1), activation=None)(x) #Reduce it down to one channel.

    model = keras.Model(inputs=image_inputs, outputs=outputs, name="detector_model")
    return model

