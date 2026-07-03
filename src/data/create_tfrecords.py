import tensorflow as tf
from pathlib import Path
import numpy as np
import cv2
YOLO_GRID_SIZE = (13,13)

def load_image_and_label(img_dir):
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

def create_example(img, label):
    img_uint8 = tf.cast(img * 255, tf.uint8)
    img_bytes = tf.io.encode_jpeg(img_uint8).numpy()
    label_bytes = label.tobytes()
    
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_bytes]))
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

with tf.io.TFRecordWriter("../../data/custom/detection.tfrecord") as writer:
    img_paths = list(Path("../../data/custom/images").glob("*.jpg"))
    for img_path in img_paths:
        img, label = load_image_and_label(str(img_path))
        example = create_example(img, label)
        writer.write(example.SerializeToString())