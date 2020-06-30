#!/usr/bin/env python


import numpy as np

import tensorflow as tf
import keras

from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.merge import add
from keras import regularizers
from keras.optimizers import Adam

tf.logging.set_verbosity(tf.logging.ERROR)
from keras.preprocessing.image import ImageDataGenerator


#======================================Data generation=============================================

train_datagenerator = ImageDataGenerator(rescale=1./255)
test_datagenerator = ImageDataGenerator(rescale=1./255)

# I didn't add path, so folders "train" and "test" should be under default directory.

train_datagenerator = train_datagenerator.flow_from_directory(
    'train',
    target_size=(128,128),
    batch_size=40,
    class_mode='binary')

test_datagenerator = test_datagenerator.flow_from_directory(
    'test',
    target_size=(128,128),
    batch_size=10,
    class_mode='binary')


#===========================================Model==================================================


def resnet8(img_width, img_height, img_channels, output_dim, prob):
    
    # Input
    img_input = Input(shape=(img_height, img_width, img_channels))
    
    x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same')(img_input)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)

    # First residual block
    x2 = keras.layers.normalization.BatchNormalization()(x1)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x2)

    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x2)

    x1 = Conv2D(32, (1, 1), strides=[2,2], padding='same')(x1)
    x3 = add([x1, x2])

    # Second residual block
    x4 = keras.layers.normalization.BatchNormalization()(x3)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x4)

    x4 = keras.layers.normalization.BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x4)

    x3 = Conv2D(64, (1, 1), strides=[2,2], padding='same')(x3)
    x5 = add([x3, x4])

    # Third residual block
    x6 = keras.layers.normalization.BatchNormalization()(x5)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x6)

    x6 = keras.layers.normalization.BatchNormalization()(x6)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x6)

    x5 = Conv2D(128, (1, 1), strides=[2,2], padding='same')(x5)
    x7 = add([x5, x6])

    x = Flatten()(x7)
    x = Activation('relu')(x)
    x = Dropout(prob)(x)

    # Angular V channel FC---regression

    # AngularV_pred= Dense(output_dim)(x)  
		
		#I didn't find regression datasets that feed images as input and 1 result as output.
 
    # Collision channel FC---classifation

    coll = Dense(output_dim)(x)
    coll = Activation('sigmoid')(coll) # sigmoid is good for binary classification


    # Define model
    model = Model(inputs=[img_input],outputs=[coll])
    #print(model.summary())
    
    return model   
#======================================train and test=======================================================

model=resnet8(128,128,3,1,0.8) # 128x128,RGB,output result, hold probability

adam=Adam(lr=0.001)
model.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])
print('Training-------------------------------------------------------------------')

DESIRED_ACCURACY = 0.80

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if((logs.get('acc')>DESIRED_ACCURACY) and (logs.get('val_acc')>DESIRED_ACCURACY )):
      print("\nReached 80% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback() 

# the accuracy stays at 70% for some reasons. Might because of RGB or loss function or hold probability is 80% for both traning and testing, so 20% chance to drop nodes even for testing. 

model.fit_generator(
    train_datagenerator,
    epochs=100,
    validation_data = test_datagenerator,
    callbacks = [callbacks]
    )

# Here is another way to train and test.

	#model=resnet8(28,28,1,1,0.5)  
	#newmodel=resnet8(28,28,1,1,1)	this way we can control hold_prob for testing

	#adam=Adam(lr=1e-5)
	#model.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])

	#print('Training-------------------------------------------------------------------')
	#model.fit(X_train_data, y_train_data, epochs=1, batch_size=64,)


	#model.save_weights('my_model_weights.h5',overwrite=True)
	#newmodel.load_weights('my_model_weights.h5')
	#newmodel.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])


	#print('\nTesting ------------------------------------------------------------------')
	#loss, accuracy= newmodel.evaluate(X_test_data, y_test_data)

	#print('\ntest loss: ', loss)
	#print('\ntest accuracy: ', accuracy)










