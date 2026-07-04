import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Turns off oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Turns off all TensorFlow C++ logs (0 = all, 1 = info, 2 = warnings, 3 = errors only)

import numpy as np
import tensorflow as tf
import cv2
from data.preprocess import preprocess_for_classifier, preprocess_for_detector
from training.loss import yolo_loss, iou_metric


classifier = tf.keras.models.load_model("../models/classifier_best.keras")
detector = tf.keras.models.load_model("../models/detector_v3_iou.keras",
    custom_objects={"yolo_loss": yolo_loss, "iou_metric": iou_metric})
cam = cv2.VideoCapture(0)

frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

def decode_predictions(output, frame_width, frame_height, threshold=0.5):
    boxes = []
    output = output[0]  # remove batch dimension → (8, 8, 5)
    for row in range(8):
        for col in range(8):
            objectness = 1 / (1 + np.exp(-output[row, col, 0]))  # sigmoid using numpy
            if objectness > threshold:
                x = output[row, col, 1]
                y = output[row, col, 2]
                w = output[row, col, 3]
                h = output[row, col, 4]
                x1 = int((x - w/2) * frame_width)
                y1 = int((y - h/2) * frame_height)
                x2 = int((x + w/2) * frame_width)
                y2 = int((y + h/2) * frame_height)
                boxes.append((x1, y1, x2, y2, objectness))
    return boxes

while True:
    ret, frame = cam.read()

    processed = preprocess_for_classifier(frame)
    detector_processed = preprocess_for_detector(frame)
    prediction = classifier.predict(processed)
    detector_prediction = detector(detector_processed, training=False).numpy()
    
    boxes = decode_predictions(detector_prediction, frame_width, frame_height)

    class_id = None
    for box in boxes:
        x1, y1, x2, y2, objectness = box        
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:  # make sure crop is valid
            processed_crop = preprocess_for_classifier(crop)
            prediction = classifier(processed_crop, training=False).numpy()
            class_id = np.argmax(prediction)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(class_id), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(
        frame, # image
        f"Predicted number of digits: {class_id}", #Text
        (5,20), # Position (Pixels from top left)
        cv2.FONT_HERSHEY_SIMPLEX, # Font
        1.0, #Font scale
        (0,255,0), # Color
        2, # Thickness
    )

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
