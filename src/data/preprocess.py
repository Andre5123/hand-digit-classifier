#Libraries
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Turns off oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Turns off all TensorFlow C++ logs (0 = all, 1 = info, 2 = warnings, 3 = errors only)

import numpy as np
import tensorflow as tf
import cv2
import random
from  pathlib import Path

IMG_SIZE_DETECTOR = (256,256,3)
IMG_SIZE_CLASSIFIER = (128,128,3)

YOLO_GRID_SIZE = (8,8)

def load_single_classification_image(img_dir, label): # Loading function so that the dataset doesn't have to store all of the images at once
    img_dir = img_dir.numpy().decode("utf-8")
    img = cv2.imread(str(img_dir))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img/255.0).astype(np.float32)

    #Random augmentation
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.clip_by_value(img, 0.0, 1.0) # Prevent values from being negative or > 1

    return img, label

def make_custom_classification_dataset(paths, labels, batch_size): #A list of the str paths and a list of the int labels
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(lambda img,label: tf.py_function(
        load_single_classification_image, [img,label], [tf.float32, tf.int32]
    ))
    dataset = dataset.map(lambda img, label: (
        tf.ensure_shape(img, IMG_SIZE_CLASSIFIER),
        tf.ensure_shape(label, ()),
    ))
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    return dataset


def make_custom_classification_datasets(crops_dir:str, batch_size:int, train_proportion:float):
    crops_dir = Path(crops_dir)
    digits_tuples = []
    for i in range(6): # Digits 0 to 5
        digit_dir = crops_dir / str(i)
        print(digit_dir)
        for img_dir in digit_dir.glob("*.jpg"):
            digits_tuples.append((img_dir, i))
    random.seed(42)
    random.shuffle(digits_tuples)
    num_imgs = len(digits_tuples)

    train_size = int(train_proportion*num_imgs)
    val_size = int((num_imgs-train_size)/2)
    test_size = num_imgs - train_size - val_size

    paths = [str(t[0]) for t in digits_tuples]
    labels = [t[1] for t in digits_tuples]

    train_dataset = make_custom_classification_dataset(paths[:train_size], labels[:train_size], batch_size)
    val_dataset = make_custom_classification_dataset(paths[train_size:train_size+val_size], labels[train_size:train_size+val_size], batch_size)
    test_dataset = make_custom_classification_dataset(paths[train_size+val_size:], labels[train_size+val_size:], batch_size)
    
    return train_dataset, val_dataset, test_dataset, train_size, val_size, test_size

def preprocess_for_classifier(img):
     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     img = cv2.resize(img, (IMG_SIZE_CLASSIFIER[0], IMG_SIZE_CLASSIFIER[1]))
     img = (img/255.0).astype(np.float32)
     img = np.expand_dims(img, axis=0)
     return img
     

# ----------------------------
# Detection dataset functions

def preprocess_for_detector(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE_DETECTOR[0], IMG_SIZE_DETECTOR[1]))
    img = (img/255.0).astype(np.float32)
    img = np.expand_dims(img, axis=0)  # add batch dim → (1, 128, 128, 3)
    return img

def parse_detection_example(example_proto, augment=False):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string)
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    
    img = tf.io.decode_jpeg(parsed['image'], channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    
    label = tf.io.decode_raw(parsed['label'], tf.float32)
    label = tf.reshape(label, (8, 8, 5))
    
    if augment:
        # Horizontal flip
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_left_right(img)
            objectness = label[..., 0:1]
            x = 1.0 - label[..., 1:2]
            ywh = label[..., 2:]
            label = tf.concat([objectness, x, ywh], axis=-1)
        
        # Brightness and contrast
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.7, 1.3)
        img = tf.image.random_hue(img, 0.1)
        img = tf.clip_by_value(img, 0.0, 1.0)
    
    return img, label

def make_tf_records_detection_datasets(tfrecord_path, batch_size, train_proportion):
    num_imgs = sum(1 for _ in tf.data.TFRecordDataset(tfrecord_path))
    dataset = tf.data.TFRecordDataset(tfrecord_path) # Take the dataset out of a tfrecord file because the images too large to process on the fly
    dataset = dataset.shuffle(num_imgs, seed=42, reshuffle_each_iteration=False)
    

    train_size = int(train_proportion*num_imgs)
    val_size = int((num_imgs-train_size)/2)
    test_size = num_imgs-train_size-val_size
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size+val_size)

    train_dataset = train_dataset.map(lambda x: parse_detection_example(x, augment=True))
    val_dataset = val_dataset.map(lambda x: parse_detection_example(x, augment=False))
    test_dataset = test_dataset.map(lambda x: parse_detection_example(x, augment=False))

    train_dataset = train_dataset.repeat().batch(batch_size)
    val_dataset = val_dataset.repeat().batch(batch_size)
    test_dataset = test_dataset.repeat().batch(batch_size)
    return train_dataset, val_dataset, test_dataset, train_size, val_size, test_size
    
    


# DEPRECATED BEYOND THIS POINT
# -------------------------------
# Classification dataset functions

def load_single_classification(img_dir):
    img_dir = img_dir.numpy().decode("utf-8")
    img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
    img = (img/255.0).astype(np.float32)
    img = np.expand_dims(img, axis=-1)
   
    label = int(img_dir[-6]) #The id is inscribed in the image name E.G [...]5R.png --> 5
    return img, label


def make_classification_dataset(data_dir, batch_size):
    dataset_path = Path(data_dir)

    img_paths = dataset_path.glob("*.png")
    img_paths = [str(x) for x in img_paths]
    num_images = len(img_paths)
    print(f"Number of images found: {num_images}")

    dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    dataset = dataset.map(lambda x: tf.py_function(
        load_single_classification,
        [x],
        [tf.float32, tf.float32]
    ))
    
    dataset = dataset.map(lambda img, label: (
        tf.ensure_shape(img, (IMG_SIZE_CLASSIFIER[0],IMG_SIZE_CLASSIFIER[1],3)),
        tf.ensure_shape(label, ()),
    ))
    
    dataset = dataset.shuffle(1000)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    return dataset, num_images

