#!/usr/bin/env python

import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D



def CNN(img_width, img_height, img_channels):

	model=Sequential()
	model.add(Conv2D(32,(4,4),activation="relu",input_shape=(img_width,img_height,img_channels)))
	model.add(MaxPooling2D(pool_size=(2,2)))


	model.add(Conv2D(32,(4,4),activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))


	model.add(Conv2D(32,(4,4),activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))


	model.add(Conv2D(32,(4,4),activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))


	model.add(Flatten())
	model.add(Dense(200))
	model.add(Dense(1))
	model.add(Activation('linear'))


	print(model.summary())

	return model

model=CNN(240,320,1)
tf.keras.utils.plot_model(model, to_file="model.png",show_layer_names=True,show_shapes=True,rankdir="TB")
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
