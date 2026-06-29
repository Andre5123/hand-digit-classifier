import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Turns off oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Turns off all TensorFlow C++ logs (0 = all, 1 = info, 2 = warnings, 3 = errors only)

import numpy as np
import tensorflow as tf
import cv2
from data.preprocess import preprocess_for_classifier

model = tf.keras.models.load_model("../models/classifier_best.keras")

cam = cv2.VideoCapture(0)

frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width,frame_height))

while True:
    ret, frame = cam.read()
    out.write(frame)

    processed = preprocess_for_classifier(frame)
    prediction = model.predict(processed)
    class_id = np.argmax(prediction)

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
out.release()
cv2.destroyAllWindows()
