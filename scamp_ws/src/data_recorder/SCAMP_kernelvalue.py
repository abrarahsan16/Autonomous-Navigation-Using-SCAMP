#!/usr/bin/env python

#import vis
import rospy
import numpy as np
import tensorflow as tf
import SCAMP_CNNmodel

model=SCAMP_CNNmodel.CNN(256,256,1,2)
model.load_weights('workwell.h5') #change according to need

#---------conv2d_1 (Conv2D)------
c1=model.layers[1]
weights1=c1.get_weights()
kernel1 = weights1[0]
print(c1)
for k in kernel1:
	print(k)
	print("-----------------------------------")
print("==================================,\n\n")
#---------conv2d_2 (Conv2D)------
c2=model.layers[3]
weights2=c2.get_weights()
kernel2 = weights2[0]
print(c2)
for k in kernel2:
	print(k)
	print("-----------------------------------")
print("==================================,\n\n")

#---------conv2d_3 (Conv2D)------
c3=model.layers[6]
weights3=c3.get_weights()
kernel3 = weights3[0]
print(c3)
for k in kernel3:
	print(k)
	print("-----------------------------------")
print("==================================,\n\n")

#---------conv2d_4 (Conv2D)------
c4=model.layers[8]
weights4=c4.get_weights()
kernel4 = weights4[0]
print(c4)
for k in kernel4:
	print(k)
	print("-----------------------------------")
print("==================================")


