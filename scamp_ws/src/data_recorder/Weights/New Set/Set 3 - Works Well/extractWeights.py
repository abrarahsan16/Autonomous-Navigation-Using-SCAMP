#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import keras

from keras.models import Model
from matplotlib import pyplot as plt
from matplotlib.pyplot import draw
from keras.models import Model

from keras.models import Sequential

import SCAMP_CNNmodel

model=SCAMP_CNNmodel.CNN(256,256,1,2)

model.load_weights('my_model_weights.h5') #change according to need
graph = tf.get_default_graph()
print("weight loaded")

for layer in model.layers:
    weights = layer.get_weights()

    print(weights)
    print("----------------")
