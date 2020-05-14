# -*- coding: utf-8 -*-
"""
Created on Thu May 14 06:53:04 2020

@author: jayar
"""

import lenet_cnn_model
import numpy as np
import cv2
import pickle

# webcam initialization using opensv, with width 640, height 640 and brightness 160
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 640)
cam.set(10, 160)


file = open('model.p', 'rb+')
model = pickle.load(file)
file.close()

while True:
    success, image = cam.read()
    img = np.asarray(image)
    img = cv2.resize(img, (64,64))
    img = lenet_cnn_model.preprocessing(img)
    img = img.reshape(1,64,64,1) # one picture with 64X64 and 1 channel
    class_prediction = int(model.predict_classes(img))
    prediction = model.predict(img)
    prediction_prob = np.amax(prediction)
    
    # the prediction greater than 65% is displayed with the prediction and prob on the image
    if prediction_prob > 0.65: 
        cv2.putText(image, str(class_prediction)+" "+str(round(prediction_prob,2), (80,80), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 0, 0), 2)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break