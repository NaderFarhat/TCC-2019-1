from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tensorflow.python.keras.optimizers import Adam
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
import random
import cv2
import os

# dimensions of our images.
img_width, img_height = 150, 150
IMAGE_SIZE = 150

train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/valid'
data_dir = './dataset'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

data = []
labels = []

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3) , input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width,img_height),
    batch_size=batch_size)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width,img_height),
    batch_size=batch_size)

X = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)


# model.fit_generator(
#     train_datagen.flow(trainX, trainY, batch_size=batch_size),
# 	validation_data=(testX, testY), steps_per_epoch=len(trainX) // batch_size,
# 	epochs=epochs, verbose=1)

model.save('modelo.h5')

plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), X.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), X.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), X.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), X.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")






