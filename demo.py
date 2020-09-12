# import the necessary packages
import numpy as np
import time
import cv2
import os
import tflite_runtime.interpreter as tflite
from Preprocessing.Preprocessing import Preprocessing
import json
import sys

model_path = "./model.tflite"
# video_path = "D:/TA/tugas-akhir-iot-parking-outdoor/M2u00002-1.m4v"
video_path = "./test.mp4"

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

if not grabbed:
    print("No Frame")
    sys.exit()

with open('maskParam_Labtek8Timur.json', 'r') as f:
# with open('maskParam.json', 'r') as f:
    Param = json.load(f)

maskParam = []

for m in Param:
    x = []
    for p in m:
        x.append(tuple(p))
    maskParam.append(x)

pre = Preprocessing()
# pre.setImage(frame)
# pre.addMask()
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

    terisi = 0
    kosong = 0

    for idx, image in enumerate(crop):

        image = cv2.resize(image, (input_shape[1], input_shape[2]), interpolation = cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image = image / 255.
        image = np.expand_dims(image, 0)

        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])
        
        cv2.polylines(frame, np.array([mask[idx]]), True, (255, 0, 0), 1)
        
        # print(preds)
        if np.argmax(preds[0])==1:
        # if preds[0][1] < 0.90:
            cv2.putText(frame,'Kosong', mask[idx][0], font, 0.5,(180, 0, 0),2)
            cv2.putText(frame,'Kosong', mask[idx][0], font, 0.5,(255, 0, 0),1)
            kosong += 1
        else:
            cv2.putText(frame,'Isi', mask[idx][0], font, 0.5,(180, 0, 0),2)
            cv2.putText(frame,'Isi', mask[idx][0], font, 0.5,(255, 0, 0),1)
            terisi += 1
    
    cv2.putText(frame,'Kosong : ' + str(kosong), (10, 30), font, 1,(26, 29, 117),4)
    cv2.putText(frame,'Kosong : ' + str(kosong), (10, 30), font, 1,(46, 51, 201),2)

    cv2.putText(frame,'Isi : ' + str(terisi), (10, 60), font, 1,(26, 29, 117),4)
    cv2.putText(frame,'Isi : ' + str(terisi), (10, 60), font, 1,(46, 51, 201),2)

    cv2.imshow("vid", frame)

    end = time.time()
    print("[INFO] Calculation time per frame : {:.5} s".format(end - start))
    
    #exit program if 'q' pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# with open('maskParam_Labtek8Timur.json', 'w') as f:
#     json.dump(pre.getMask(), f)

# release the file pointers
print("[INFO] cleaning up...")
vs.release()
cv2.destroyAllWindows()