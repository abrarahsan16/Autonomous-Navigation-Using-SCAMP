#!/usr/bin/env python

import numpy as np
import cv2
import tensorflow as tf
import re
import os
import glob

import keras
from keras.models import Model
from matplotlib import pyplot as plt
from matplotlib.pyplot import draw

import vis
from vis.utils import utils
from vis.visualization import visualize_cam, visualize_saliency

import SCAMP_CNNmodel

folder = "/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/data_recorder/Data Collection Archive/data"
assert folder, "Provide the dataset folder"
experiment = glob.glob(folder + "/*")

outputFolder = "/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/data_recorder/Heatmap/"

model = SCAMP_CNNmodel.CNN(256,256,1,3)
model.load_weights('my_model_weights.h5') #change according to need
graph = tf.get_default_graph()
print("weight loaded")


for exp in experiment:
    outp = outputFolder
    print(exp)
    images = [os.path.basename(x) for x in glob.glob(exp + "/img/*.jpeg")]
    mapmodel = Model(inputs=model.inputs, outputs=model.layers[7].output)
    mapmodel2 = Model(inputs=model.inputs, outputs=model.layers[2].output)
    layer_idx=utils.find_layer_idx(mapmodel, 'activation_1')
    #layer_idx=utils.find_layer_idx(mapmodel, 'max_pooling2d_2')
    mapmodel.layers[layer_idx].activation = keras.activations.linear
    model2= utils.apply_modifications(mapmodel)
    for im in images:
        stamp = int(re.sub(r'\.jpeg$','',im))
        #out = resizer(exp + "/img/"+ im)
        im = exp + "/img/"+ im
        im = cv2.imread(im)
        cv2_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        cv2_res = cv2.resize(cv2_gray, dsize=(256, 256)) # needs center crop
        np_img = np.asarray(cv2_res)
        out = np_img.reshape([1,256,256,1])
        #heatMap(im, stamp)
        #cv2.imwrite(os.path.join(outp,str(stamp)+".jpeg"), out)


        #mapmodel = Model(inputs=model.inputs, outputs=model.layers[3].output)
        fm = mapmodel2.predict(out)
        y_pred=model2.predict(out)
        class_idxs_sorted =np.argsort(y_pred.flatten())[::-1]
        test=np.argsort(y_pred.flatten())
        penultimate_layer_idx = utils.find_layer_idx(model2, "conv2d_2")
        class_idx=class_idxs_sorted[0]
        seed_input=out
        grad_top1= visualize_cam(model2, layer_idx,[0],seed_input,
            penultimate_layer_idx = penultimate_layer_idx,
            backprop_modifier= None, grad_modifier= None)

        fig,axes=plt.subplots(1,3,figsize=(25,15))
        axes[0].imshow(cv2_res)
        axes[0].set_title("Cov2D Input")
        axes[1].imshow(fm[0,:,:,2], cmap='viridis')
        axes[1].set_title("Feature Map")
        axes[2].imshow(cv2_res)
        axes[2].set_title("Grad Cam")
        axes[2].imshow(grad_top1,cmap="jet",alpha=0.3)
        #fig.colorbar(i)
        addr = outputFolder + str(stamp)
        plt.savefig(addr)
        print(stamp)
        plt.close()
    print("Done")
