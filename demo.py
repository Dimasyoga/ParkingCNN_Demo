# import the necessary packages
import numpy as np
import time
import cv2
import os
import tflite_runtime.interpreter as tflite
from Preprocessing.Preprocessing import Preprocessing
import json

model_path = "./model.tflite"
video_path = "./test.mp4"
# video_path = "D:/TA/tugas-akhir-iot-parking-outdoor/M2u00002-1.m4v"

font = cv2.FONT_HERSHEY_SIMPLEX

# Load TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']

vs = cv2.VideoCapture(video_path)

(grabbed, frame) = vs.read()

with open('maskParam.json', 'r') as f:
    Param = json.load(f)

maskParam = []

for m in Param:
    x = []
    for p in m:
        x.append(tuple(p))
    maskParam.append(x)

pre = Preprocessing()
pre.setImage(frame)
pre.setMask(maskParam)

# loop over frames from the video file stream
while True:
    start = time.time()
    # read the next frame from the file
    (grabbed, frame) = vs.read()
 
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    pre.setImage(frame)
    crop = pre.getCrop()
    mask = pre.getMask()

    for idx, image in enumerate(crop):

        image = cv2.resize(image, (input_shape[1], input_shape[2]), interpolation = cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image = image / 255.
        image = np.expand_dims(image, 0)

        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])
        
        if np.argmax(preds[0])==1:
            cv2.putText(frame,'FREE', mask[idx][0], font, 1,(255,255,255),1)
        else:
            cv2.putText(frame,'BUSY', mask[idx][0], font, 1,(255,255,255),1)

        cv2.imshow("output", frame)
    
    end = time.time()
    print("[INFO] Compute time per frame : {:.5} s".format(end - start))
    
    #exit program if 'q' pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the file pointers
print("[INFO] cleaning up...")
vs.release()
cv2.destroyAllWindows()