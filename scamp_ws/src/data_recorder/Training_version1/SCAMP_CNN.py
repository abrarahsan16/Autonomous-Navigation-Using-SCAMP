#!/usr/bin/env python


import tensorflow as tf
import numpy as np
import os
#import sklearn.model_selection import train_test_split
import sys

from keras.callbacks import ModelCheckpoint
from keras import optimizers

import SCAMP_utils
import SCAMP_CNNmodel


#=====================================GET MODEL===================================================
img_width=480
img_height=640
img_channels=1
output_dim=1

model=SCAMP_CNNmodel.CNN(img_width, img_height,img_channels,output_dim)


#=====================================SET DATA BATCH================================================

train_datagen=SCAMP_utils.DroneDataGenerator(rotation_range = 0.2,
                                             rescale = 1./255,
                                             width_shift_range = 0.2,
                                             height_shift_range=0.2)

train_generator=train_datagen.flow_from_directory("/home/andrew/turtlebot3_ws/src/Converter/src/Test1/data",
                                                        shuffle = True,
                                                        color_mode='grayscale',
                                                        target_size=(480,640),
                                                        crop_size=(480,640),
                                                        batch_size = 32)

val_datagen =SCAMP_utils.DroneDataGenerator(rescale=1./255)

val_generator =val_datagen.flow_from_directory("/home/andrew/turtlebot3_ws/src/Converter/src/Test1/testfolder",
                                                        shuffle = True,
                                                        color_mode='grayscale',
                                                        target_size=(480,640),
                                                        crop_size=(480,640),
                                                        batch_size = 32)


#=====================================COMPILE AND TRAIN================================================

model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
print('Training-------------------------------------------------------------------')


DESIRED_ACCURACY = 0.60

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if((logs.get('acc')>DESIRED_ACCURACY) and (logs.get('val_acc')>DESIRED_ACCURACY )):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

model.fit_generator(train_generator,
                    epochs=100,
                    validation_data=val_generator,
		    callbacks = [callbacks])



























