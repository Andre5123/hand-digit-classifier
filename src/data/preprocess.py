#Libraries
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Turns off oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Turns off all TensorFlow C++ logs (0 = all, 1 = info, 2 = warnings, 3 = errors only)

import numpy as np
import tensorflow as tf
import cv2
import random
from  pathlib import Path

IMG_SIZE_DETECTOR = (416,416)
IMG_SIZE_CLASSIFIER = (128,128,3)

YOLO_GRID_SIZE = (13,13)

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






# ----------------------------
# Preprocessing an image for classifier testing (On live webcam)
def preprocess_for_classifier(img):
     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     img = cv2.resize(img, (IMG_SIZE_CLASSIFIER[0], IMG_SIZE_CLASSIFIER[1]))
     img = (img/255.0).astype(np.float32)
     img = np.expand_dims(img, axis=0)
     return img
     

# ----------------------------
# Detection dataset functions
def load_single_detection(img_dir):
    img_dir = img_dir.numpy().decode("utf-8")
    label = np.zeros((YOLO_GRID_SIZE[0], YOLO_GRID_SIZE[1], 5)) # 5 is for the bounding box parameters. 
    
    # Load and normalize img
    img = cv2.imread(img_dir)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img/255.0).astype(np.float32)

    with open(Path(img_dir).parent.parent / "labels" / (Path(img_dir).stem+".txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                box_values = line.strip().split()
                
                # Ignore box_values[0], that is the class ID. There is only one class, hands, which is 0.
                box_x = float(box_values[1]) # x-coord over the entire image, from 0 to 1.
                box_y = float(box_values[2]) # y-coord over the entire image, from 0 to 1.
                box_w = float(box_values[3])
                box_h = float(box_values[4])

                grid_cell_x = int(box_x*YOLO_GRID_SIZE[0]) # The indices of the grid cell that owns the bounding box
                grid_cell_y = int(box_y*YOLO_GRID_SIZE[1])

                label[grid_cell_y, grid_cell_x] = [1, box_x, box_y, box_w, box_h]
    return img, label

def make_custom_detection_dataset(img_paths, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    dataset = dataset.map(lambda x: tf.py_function( 
        load_single_detection, [x], [tf.float32, tf.float32],
    ), num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.map(lambda img, label: (
        tf.ensure_shape(img, (IMG_SIZE_DETECTOR[0],IMG_SIZE_DETECTOR[1],3)),
        tf.ensure_shape(label, (13,13,5)),
    ))
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def make_custom_detection_datasets(data_dir, batch_size, train_proportion):
    dataset_path = Path(data_dir)
    img_paths = list(dataset_path.glob("*.jpg"))
    num_imgs = len(img_paths)

    img_paths = [str(img) for img in img_paths]

    random.seed(42)
    random.shuffle(img_paths)

    train_size = int(train_proportion*num_imgs)
    val_size = int((num_imgs-train_size)/2)
    test_size = num_imgs - train_size - val_size

    train_dataset = make_custom_detection_dataset(img_paths[:train_size], batch_size)
    val_dataset = make_custom_detection_dataset(img_paths[train_size:train_size+val_size], batch_size)
    test_dataset = make_custom_detection_dataset(img_paths[train_size+val_size:], batch_size)
    
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
        tf.ensure_shape(img, (128,128,1)),
        tf.ensure_shape(label, ()),
    ))
    
    dataset = dataset.shuffle(1000)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    return dataset, num_images

