# -*- coding: utf-8 -*-
"""
Created on Thu May 14 06:48:04 2020

@author: jayar
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
import pickle
import os

images_path = os.getcwd()+r'\myData'
classes = len(range(0,10,1))
images = []
classNbr = []
nbrClasses = list(range(0,10,1))

# import images and corresponding classes
for i in nbrClasses:
    pics_list = os.listdir(images_path+f'\{i}')
    for pic in pics_list:
        img = cv2.imread(images_path+f'\{i}'+f'\{str(pic)}')
        img = cv2.resize(img, (64,64))
        images.append(img)
        classNbr.append(i)
cv2.imshow('simple image', images[45])
cv2.waitKey(0)
print(len(images), len(classNbr))

images = np.array(images)
classNbr = np.array(classNbr)
print(images.shape, classNbr.shape)

for x in range(10):
    print(f"class {x} has {len(np.where(classNbr==x)[0])}")
    
# Image processing to gray scale, blur, equalize histogram to increase intensity and normalize the array
def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

X_train,X_test,y_train,y_test = train_test_split(images,classNbr,test_size=0.2)
X_train,X_validation,y_train,y_validation = train_test_split(X_train,y_train,test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)
for x in range(10):
    print(f"class {x} has {len(np.where(y_validation==x)[0])}")
    
    
# expanding the image array to include one channel for keras training
X_train = np.array(list(map(preprocessing, X_train))).reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
X_test = np.array(list(map(preprocessing, X_test))).reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))
X_validation = np.array(list(map(preprocessing, X_validation))).reshape((X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1))


"""
Image Augmentation using keras for generating images with different params (horizontal and vertical flips were avoided)
one hot encoding of y classes
""" 

image_gen = ImageDataGenerator(width_shift_range=0.2,
                              height_shift_range=0.2,
                              brightness_range=(1,2),
                              shear_range=0.2,
                              zoom_range=0.2,
                              rotation_range=10)

image_gen.fit(X_train)
y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)
y_validation = to_categorical(y_validation, classes)


# CNN model Lenet model
def lenet_model():
    model = Sequential()
    model.add((Conv2D(60,(5,5),input_shape=(64,64,1),activation='relu')))
    model.add((Conv2D(60, (5,5), activation='relu')))
    model.add(MaxPooling2D((2,2)))
    model.add((Conv2D(30, (3,3), activation='relu')))
    model.add((Conv2D(30, (3,3), activation='relu')))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model
model = lenet_model()

print(model.summary())

training = model.fit_generator(image_gen.flow(X_train, y_train, batch_size=50),
                               steps_per_epoch=2000,
                               epochs=20,
                               validation_data=(X_validation,y_validation), 
                               shuffle=1)

pickle_model = open('model.p', 'wb+')
pickle.dump(model, pickle_model)
pickle_model.close()


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,5)) 
ax[0].plot(training.history['loss'])
ax[0].plot(training.history['val_loss'])
ax[0].legend(['training', 'validation'])
ax[0].set_title('Loss')
ax[0].set_xlabel('epochs')
ax[1].plot(training.history['accuracy'])
ax[1].plot(training.history['val_accuracy'])
ax[1].legend(['training', 'validation'])
ax[1].set_title('Accuracy')
ax[1].set_xlabel('epochs')
fig.tight_layout()


test_score = model.evaluate(X_test,y_test,verbose=1)
print(f'Test loss: {test_score[0]} and Test accuracy: {test_score[1]}')