import re
import os
import numpy as np
import tensorflow as tf
import json
import cv2
import utils

#===========================Testing use====================================================
train_datagen=utils.DataGenerator(rescale = 1./255)

train_generator=train_datagen.flow_from_directory("/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/data_recorder/data",
                                                        shuffle = True,
                                                        color_mode='grayscale',
                                                        target_size=(480,640),
                                                        crop_size=(480,640),
                                                        batch_size = 32)



val_datagen=utils.DataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory("/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/data_recorder/test",
                                                        shuffle = True,
                                                        color_mode='grayscale',
                                                        target_size=(480,640),
                                                        crop_size=(480,640),
                                                        batch_size = 32)

a,b=train_generator.next()
