# DEPRECATED
# This script collects live training data using the manual webcam method.
# Press f to freeze the recording, then click and drag to place a bounding box around a hand, then press a number to give it a class.
# Repeat if needed for the 2nd hand. When you are done press d and it will save the frame, its bounding box annotations and crops of all of the bounded hands.


import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Turns off oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Turns off all TensorFlow C++ logs (0 = all, 1 = info, 2 = warnings, 3 = errors only)

import numpy as np
import tensorflow as tf
import cv2

import time


cam = cv2.VideoCapture(0)

frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width,frame_height)) # Uncomment this to keep the mp4 file


frozen = False
countdown_start = None

image_counter = 0
with open("image_counter.txt", "r") as f:
    image_counter = int(f.read().strip())


# Webcam recording loop.
while True:
    #out.write(frame) # Uncomment this to keep the mp4 file

    ret, frame = cam.read()
    

    cmdKey = cv2.waitKey(1)

    if cmdKey == ord('q'):
        break
    elif cmdKey == ord('f'): # Freeze the current frame
        frozen = True
    elif cmdKey == ord('t'): # Timed freeze. Gives 3 second to position yourself
        countdown_start = time.time()
        
    if countdown_start is not None: # Display countdown
        elapsed = time.time() - countdown_start
        remaining = 3-int(elapsed)
        if remaining > 0:
            cv2.putText(frame, str(remaining), (frame_width//2, frame_height//2), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0,255.0,0), 4)
        if elapsed >= 3:
            frozen = True
            countdown_start = None

    cv2.imshow('Camera', frame)
    if frozen:
        image_counter +=1
        done = False

        cv2.imwrite(f"../../data/custom/images/frame_{image_counter:05d}.jpg", frame) # Save the current frame to images
        with open(f"../../data/custom/labels/frame_{image_counter:05d}.txt", "w") as f: #Open a new file for each image even if there is no info to store
            pass
        crop_number = 0
        while not done: # Collect the bounding box data for however many hands there are in the image
            
            crop_number +=1
            key = cv2.waitKey(0) #Prompt the user to input a class id for the number of hand digits.

            while key not in range(ord('0'), ord('5')+1): # input must be a digit 0-5 or 'd'.
                if key == ord("d"):
                    print("Done with crops")
                    done = True
                    break
                print("Enter a digit 0-5")
                key = cv2.waitKey(0)

            if done==True: continue

            digit = int(chr(key))
            print("This crop will be labelled", digit)

            
            roi = cv2.selectROI('Camera', frame) #User clicks and drags to put a bounding box around a hand.
            x_pixels = roi[0]
            y_pixels = roi[1]
            w_pixels = roi[2]
            h_pixels = roi[3]
            print("Box dimensions", x_pixels, y_pixels, w_pixels, h_pixels)
            
            x = (x_pixels+w_pixels/2)/frame_width
            y = (y_pixels+h_pixels/2)/frame_height
            w = (w_pixels)/frame_width
            h = (h_pixels)/frame_height


            with open(f"../../data/custom/labels/frame_{image_counter:05d}.txt", "a") as f:
                f.write(f"0 {x} {y} {w} {h}\n")
            crop = frame[y_pixels:y_pixels+h_pixels, x_pixels:x_pixels+w_pixels]
            crop = cv2.resize(crop, (128,128))
            cv2.imwrite(f"../../data/custom/crops/{digit}/crop_{image_counter:05d}-{crop_number}.jpg", crop)

        frozen = False
        
        
with open("image_counter.txt", "w") as f:
    f.write(str(image_counter))
cam.release()
#out.release() # Uncomment this to keep the mp4 file
cv2.destroyAllWindows()
