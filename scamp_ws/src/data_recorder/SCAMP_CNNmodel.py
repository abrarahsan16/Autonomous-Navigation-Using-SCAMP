#!/usr/bin/env python

import os
#os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


def CNN(img_width, img_height, img_channels, output_dim):

	model=Sequential()
	model.add(Conv2D(3,(4,4),activation="relu",input_shape=(img_width,img_height,img_channels)))
	model.add(MaxPooling2D(pool_size=(2,2)))


	model.add(Conv2D(3,(4,4),activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))


	#model.add(Conv2D(32,(4,4),activation="relu"))
	#model.add(MaxPooling2D(pool_size=(2,2)))


	#model.add(Conv2D(32,(4,4),activation="relu"))
	#model.add(MaxPooling2D(pool_size=(2,2)))


	model.add(Flatten())
	model.add(Dense(200))
	model.add(Dense(output_dim))
	model.add(Activation('softmax'))


	print(model.summary())
	tf.keras.utils.plot_model(model, to_file="model.png",show_layer_names=True,show_shapes=True,rankdir="TB")
	return model
