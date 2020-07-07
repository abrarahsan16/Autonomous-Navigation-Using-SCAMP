import tensorflow as tf
import numpy as np
import os
#import sklearn.model_selection import train_test_split
import sys

from keras.callbacks import ModelCheckpoint
from keras import optimizers

import utils
import SCAMP_CNNmodel

def getModel(img_width, img_height, img_channels):
    model = SCAMP_CNNmodel.CNN(img_width, img_height, img_channels)

    if weights_path:
        try:
            model.load_weights(weights_path)
            print("Loaded model")
        except:
            print("Cannot find weight path.")

    return model


def trainModel(train_data_generator, model, initial_epoch):

    #Loss weights
    model.alpha = tf.Variable(1,trainable=False, name="alpha", dtype=tf.float32)
    model.beta = tf.Variable(0,trainable=False, name="alpha", dtype=tf.float32)

    #Number of samples for hard_mining
    model.k_mse = tf.Variables(batch_size=32, trainable=False, name="k_mse", dtype=tf.int32)
    model.entropy = tf.Variables(batch_size=32, trainable=False, name="k_mse", dtype=tf.int32)

    optimizer = optimizers.Adam(decay=1e-5)

    model.compile(loss=[utils.hard_mining_mse(model.k_mse),utils.hard_mining_entropy(model.k_entropy)],optimizer=optimizer, loss_weights=[model.alpha, model.beta])

    #Save model with the lowest validation loss_weights
    weights_path = os.path.join("/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/logs/",'weights_{epohc:03d}.h5')
    writeBestModel = ModelCheckpoint(filepath=weights_path, monitor='val_loss',save_best_only=True, save_weights_only=True)

    # Save model every 'log_rate' epochs.
    # Save training and validation losses.
    #logz.configure_output_dir("/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/logs/")
    #saveModelAndLoss = log_utils.MyCallback("/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/logs/", period=10,batch_size=32)

    steps_per_epoch = int(np.ceil(train_data_generator.samples / 32))
    validation_steps = int(np.ceil(val_data_generator.samples / 32))

    model.fit_generator(train_data_generator,
                        epochs=100, steps_per_epoch = steps_per_epoch,
                        callbacks=[writeBestModel, saveModelAndLoss],
                        validation_data=val_data_generator,
                        validation_steps = validation_steps,
                        initial_epoch=initial_epoch)

def _main():

    # Create the experiment rootdir if not already there
    if not os.path.exists("/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/logs"):
        os.makedirs("/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/logs")

    # Input image dimensions
    img_width, img_height = 320, 240

    # Cropped image dimensions
    crop_img_width, crop_img_height = 250, 250

    # Image mode
    img_channels = 3

    # Output dimension (one for steering and one for collision)
    output_dim = 1

    # Generate training data with real-time augmentation
    train_datagen = utils.DataGenerator(rotation_range = 0.2,
                                             rescale = 1./255,
                                             width_shift_range = 0.2,
                                             height_shift_range=0.2)

    train_generator = train_datagen.flow_from_directory("/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/data",
                                                        shuffle = True,
                                                        color_mode='grayscale',
                                                        target_size=(img_width, img_height),
                                                        crop_size=(crop_img_height, crop_img_width),
                                                        batch_size = 32)

    # Weights to restore
    weights_path = os.path.join("/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/logs", "model_weights.h5")
    initial_epoch = 0
    if not weights_path:
        # In this case weights will start from random
        weights_path = None
    else:
        # In this case weigths will start from the specified model
        initial_epoch = 0

    # Define model
    model = getModel(crop_img_width, crop_img_height, img_channels,
                        output_dim, weights_path)

    # Serialize model into json
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    utils.modelToJson(model, json_model_path)

    # Train model
    trainModel(train_generator, val_generator, model, initial_epoch)
