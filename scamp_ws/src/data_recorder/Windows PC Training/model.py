#!/usr/bin/env python

import os
#os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow as tf
import keras

from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.merge import add
from keras import regularizers

def CNN(img_width, img_height, img_channels, output_dim):
    img_input = Input(shape=(img_height, img_width, img_channels))
    x1 = Conv2D(2, (3, 3), strides = [2,2], padding='same')(img_input)
    x1 = Conv2D(1, (3, 3), strides = [2,2], padding='same')(x1)
    x1 = Conv2D(2, (3, 3), strides = [2,2], padding='same')(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)
    x = Flatten()(x1)
    x = Dense(100)(x)
    steer=Dense(output_dim)(x)
    steer=Activation('softmax')(steer)

    coll=Dense(output_dim)(x)
    coll = Activation('softmax')(coll)

    model = Model(inputs=[img_input], outputs=[steer, coll])
    print(model.summary())
    return model