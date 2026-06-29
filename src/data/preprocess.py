#Libraries
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Turns off oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Turns off all TensorFlow C++ logs (0 = all, 1 = info, 2 = warnings, 3 = errors only)

import numpy as np
import tensorflow as tf
import cv2
from  pathlib import Path

IMG_SIZE_DETECTOR = (416,416)
IMG_SIZE_CLASSIFIER = (128,128)

YOLO_GRID_SIZE = (19,19)

# ----------------------------
# Preprocessing an image for classifier testing (On live webcam)
def preprocess_for_classifier(img):
     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     img = cv2.resize(img, IMG_SIZE_CLASSIFIER)
     img = (img/255.0).astype(np.float32)
     img = np.expand_dims(img, axis=-1)
     img = np.expand_dims(img, axis= 0)
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

    with open(Path(img_dir).with_suffix(".txt"), "r") as f:
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

def make_detection_dataset(data_dir, batch_size):
    dataset_path = Path(data_dir)
    img_paths = dataset_path.glob("*.jpg")
    num_images = len(img_paths)

    img_paths = [str(img) for img in img_paths]

    dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    dataset = dataset.map(lambda x: tf.py_function( 
        load_single_detection, [x], [tf.float32, tf.float32] #
    ))
    dataset = dataset.shuffle(1000)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    return dataset, num_images
    



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
    print(dataset_path.resolve(), "is the resolved path")

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

