import numpy as np
import os
#os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sys
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import optimizers

import utils
import SCAMP_CNNmodel

img_width=480
img_height=640
img_channels=1
output_dim=3
b_size = 3
crop_width = 256
crop_height = 256
def combine_generator(gen1, gen2):
    while True:
        yield(gen1.next(),gen2.next())

model = SCAMP_CNNmodel.CNN(crop_width,crop_height,img_channels,output_dim)
print("Loaded model")

train_datagen = utils.DataGenerator(rescale = 1./255)
print("Started Data Collection")
train_generator = train_datagen.flow_from_directory("/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/data_recorder/data",
        shuffle=True, color_mode='grayscale', target_size=(480,640), crop_size=(crop_height,crop_width), batch_size=b_size)

val_generator = train_datagen.flow_from_directory("/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/data_recorder/test/",
        shuffle=True, color_mode='grayscale', target_size=(480,640), crop_size=(crop_height,crop_width), batch_size=b_size)
#xTrain, xTest, yTrain, yTest = train_test_split(train_generator, test_size=0.4, random_state=0)

combgen = combine_generator(train_generator,val_generator)
opt = optimizers.Adam(lr=0.00001)

#model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
print("Training..................................................................")


class myCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.acc= []
        self.val_losses = []
        self.val_acc = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        # rewrite on_epoch_end, to plt at the end of each epoch,
        #if things go too off, we dont have to wait until the end to kill it and adjust
        # I'm not sure if there is a function that can be called every step of training.

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        self.i+= 1

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:

            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            # print(plt.style.available) to see the available options
            #plt.style.use("seaborn")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            #plt.figure()
            #plt.show(block=False)
            plt.plot(N, self.losses, label = "train_loss")
            plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_losses, label = "val_loss")
            plt.plot(N, self.val_acc, label = "val_acc")

            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            #plt.show(block=False)
            plt.pause(10) # wait for 10 sec and then close the figure so the training can continue.
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            plt.savefig('Plot Output/Epoch-{}.png'.format(epoch+1))
            plt.close()
            model.save_weights('my_model_weights.h5',overwrite=True) # I didn't define path, so it should be stored in default path. For me it's home/

callbacks = myCallback()

tf.keras.utils.plot_model(model, to_file="model.png",show_layer_names=True,show_shapes=True,rankdir="TB")
history=model.fit_generator(train_generator,steps_per_epoch=int(train_generator.samples/b_size), validation_data=val_generator,validation_steps=int(val_generator.samples/b_size),max_queue_size=10, epochs=10, verbose=1,callbacks = [callbacks],workers=5)
#history=model.fit_generator(train_generator,steps_per_epoch=int(train_generator.samples/b_size),max_queue_size=10, epochs=20, verbose=1,callbacks = [callbacks],workers=5)

# Final acc and loss graph when all trainings are done
#loss_train = history.history['loss']
#acc_train = history.history['acc']
#plt.plot(loss_train, 'g', label='Training loss')
#plt.plot(acc_train, 'b', label='Training accuracy')
#plt.title('This is a title')
#plt.xlabel('Epochs')
#plt.ylabel('this is y label')
#plt.legend()
#plt.show()



model.save_weights('my_model_weights.h5',overwrite=True) # I didn't define path, so it should be stored in default path. For me it's home/

#model2 = SCAMP_CNNmodel.CNN(crop_width,crop_height,img_channels,output_dim) #define new model that has the same structure
#model2.load_weights('my_model_weights.h5')
#model2.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy']) #compile new model that has trained weight loaded.
# train again...
# I haven't tested yet if this way would work.
