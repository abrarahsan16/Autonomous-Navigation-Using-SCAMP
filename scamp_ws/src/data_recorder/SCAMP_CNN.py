import numpy as np
import os
#os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sys

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import optimizers

import utils
import SCAMP_CNNmodel

img_width=480
img_height=640
img_channels=1
output_dim=1

def combine_generator(gen1, gen2):
    while True:
        yield(gen1.next(),gen2.next())

model = SCAMP_CNNmodel.CNN(img_width,img_height,img_channels,output_dim)
print("Loaded model")

train_datagen = utils.DataGenerator(rescale = 1./255)
print("Started Data Collection")
train_generator = train_datagen.flow_from_directory("/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/data_recorder/data",
        shuffle=True, color_mode='grayscale', target_size=(480,640), crop_size=(480,640), batch_size=32)

val_generator = train_datagen.flow_from_directory("/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/data_recorder/test/",
        shuffle=True, color_mode='grayscale', target_size=(480,640), crop_size=(480,640), batch_size=32)
#xTrain, xTest, yTrain, yTest = train_test_split(train_generator, test_size=0.4, random_state=0)

combgen = combine_generator(train_generator,val_generator)
opt = optimizers.Adam(lr=0.01)

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
print("Training..................................................................")

DESIRED_ACCURACY = 0.60

class callBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if((logs.get("Acc")>DESIRED_ACCURACY and (logs.get('val_acc')>DESIRED_ACCURACY) )):
            print("\nReached 60% accuracy so cancelling training!")

callbacks = callBack()

model.fit_generator(train_generator,steps_per_epoch=500,max_queue_size=5, epochs=10, verbose=1,callbacks = [callbacks],workers=3, validation_data=val_generator,validation_steps=int(validation_generator.samples/32))
