#!/usr/bin/env python
import numpy as np
from decimal import *
import time
import random
import glob
import csv
import sys
import re
import os
import cv2
import rospy
import numpy as np
import cv2
import tensorflow as tf
import statistics as st

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from keras.models import Model
import keras

bridge = CvBridge()

turn = 0
linear = 0
count=0
maxturn=""
l1=[]
l2=[]
l3=[]
l4=[]

import SCAMP_CNNmodel


def sneg(register):
    temp_list = []
    row_counter = 0
    for row in register:
        temp_row = []
        item_counter = 0
        for item in row:
            # Check FLAG
            if flag[row_counter][item_counter] == 1:
                # temp_row.append(-item)
                temp_row.append(-check_bounds(random_error(item)))
            else:
                temp_row.append(item)
            item_counter += 1
        temp_list.append(temp_row)
        row_counter += 1
    return temp_list
def div2(register):
    temp_list = []
    row_counter = 0
    for row in register:
        temp_row = []
        item_counter = 0
        for item in row:
            if flag[row_counter][item_counter] == 1:
                # temp_row.append(item/2)
                temp_row.append(check_bounds(random_error(div2_check(item))))
                # temp_row.append(div2_check(check_bounds(random_error(item/2))))
            else:
                temp_row.append(item)
            item_counter += 1
        temp_list.append(temp_row)
        row_counter += 1
    return temp_list
def div9(register):
    temp_list = []
    row_counter = 0
    for row in register:
        temp_row = []
        item_counter = 0
        for item in row:
            if flag[row_counter][item_counter] == 1:
                temp_row.append(item/9)
            else:
                temp_row.append(item)
            item_counter += 1
        temp_list.append(temp_row)
        row_counter += 1
    return temp_list
def add(register1, register2):
    temp_list = []
    row_counter = 0
    for row in register1:
        temp_row = []
        item_counter = 0
        for item in row:
            if flag[row_counter][item_counter] == 1:
                register1_value = item
                register2_value = register2[row_counter][item_counter]
                # temp_row.append(register1_value + register2_value)
                temp_row.append(check_bounds(random_error(register1_value + register2_value)))
            else:
                temp_row.append(item)
            item_counter += 1
        temp_list.append(temp_row)
        row_counter += 1
    return temp_list
def sub(register1, register2):
    temp_list = []
    row_counter = 0
    for row in register1:
        temp_row = []
        item_counter = 0
        for item in row:
            if flag[row_counter][item_counter] == 1:
                register1_value = item
                register2_value = register2[row_counter][item_counter]
                # temp_row.append(register1_value - register2_value)
                temp_row.append(check_bounds(random_error(register1_value - register2_value)))
            else:
                temp_row.append(item)
            item_counter += 1
        temp_list.append(temp_row)
        row_counter += 1
    return temp_list
def north(register):
    temp_list = []
    row_counter = 0
    for row in register:
        # Deal with overflow
        if row_counter == 0:
            temp_row = []
            item_counter = 0
            for item in row:
                if flag[row_counter][item_counter] == 1:
                    temp_row.append(0)
                else:
                    temp_row.append(item)
                item_counter += 1
            temp_list.append(temp_row)
            row_counter += 1
            continue

        temp_row = []
        item_counter = 0
        for item in row:
            if flag[row_counter][item_counter] == 1:
                register2_value = register[row_counter-1][item_counter]
                # temp_row.append(register2_value)
                temp_row.append(check_bounds(random_error(register2_value)))
            else:
                temp_row.append(item)
            item_counter += 1
        temp_list.append(temp_row)
        row_counter += 1
    return temp_list
def south(register):
    temp_list = []
    row_counter = 0
    for row in register:
        # Deal with overflow
        if row_counter == (len(register) - 1):
            temp_row = []
            item_counter = 0
            for item in row:
                if flag[row_counter][item_counter] == 1:
                    temp_row.append(0)
                else:
                    temp_row.append(item)
                item_counter += 1
            temp_list.append(temp_row)
            row_counter += 1
            continue

        temp_row = []
        item_counter = 0
        for item in row:
            if flag[row_counter][item_counter] == 1:
                register2_value = register[row_counter+1][item_counter]
                # temp_row.append(register2_value)
                temp_row.append(check_bounds(random_error(register2_value)))
            else:
                temp_row.append(item)
            item_counter += 1
        temp_list.append(temp_row)
        row_counter += 1
    return temp_list
def east(register):
    temp_list = []
    row_counter = 0
    for row in register:
        temp_row = []
        item_counter = 0
        for item in row:
            # Deal with overflow
            if item_counter == (len(row) - 1):
                if flag[row_counter][item_counter] == 1:
                    temp_row.append(0)
                else:
                    temp_row.append(item)
                item_counter += 1
                continue

            if flag[row_counter][item_counter] == 1:
                register2_value = register[row_counter][item_counter+1]
                temp_row.append(check_bounds(random_error(register2_value)))
            else:
                temp_row.append(item)
            item_counter += 1
        temp_list.append(temp_row)
        row_counter += 1
    return temp_list
def west(register):
    temp_list = []
    row_counter = 0
    for row in register:
        temp_row = []
        item_counter = 0
        for item in row:
            # Deal with overflow
            if item_counter == 0:
                if flag[row_counter][item_counter] == 1:
                    temp_row.append(0)
                else:
                    temp_row.append(item)
                item_counter += 1
                continue
            if flag[row_counter][item_counter] == 1:
                register2_value = register[row_counter][item_counter-1]
                temp_row.append(check_bounds(random_error(register2_value)))
            else:
                temp_row.append(item)
            item_counter += 1
        temp_list.append(temp_row)
        row_counter += 1
    return temp_list
def WHERE(register):
    global flag
    flag = list(register)
def ALL():
    global flag
    flag = []
    for i in range(256):
      temp_row = []
      for i in range(256):
        temp_row.append(1)
      flag.append(temp_row)

def relu(register):
    temp_list = []
    for row in register:
        temp_row = []
        for item in row:
            if item > 0:
                temp_row.append(item)
            else:
                temp_row.append(0)
        temp_list.append(temp_row)
    return temp_list

def check_bounds(value):
    if value > 256.0:
        return 256.0
    elif value < -256.0:
        return -256.0
    return value
def random_error(value):
    multiplier = random.uniform(0.95, 0.99)
    return multiplier*value
def div2_check(value):
    value = float(value)
    if (value == 0):
        return 0
    if (value < 20.0) and (value > 0):
        return value/1.5
    if (value > -20.0) and (value < 0.0):
        return value/1.5
    return value/2

def stride2(register):
    register=np.array(register)
    temp=np.zeros(shape=(int(register.shape[0]/2),int(register.shape[0]/2)))
    i=0
    j=0
    I=0
    J=0
    while i<register.shape[0]:
        while j<register.shape[0]:
            temp[I][J]=register[i][j]
            j=j+2
            J=J+1
        j=0
        J=0
        i=i+2
        I=I+1
    return temp.tolist()

def highlight(input):
    A=input
    for i in range(64):
        for j in range(64):
            if A[i][j]>15:
                A[i][j]=250
            else:
                A[i][j] =0
    return A

flag = []
for i in range(256):  # 256 in our case
  temp_row = []
  for i in range(256):   # 256 in our case
    temp_row.append(1)
  flag.append(temp_row)

def conv1_kernel1(input):
    A=input

    A = north(A)
    A = div2(A)
    A = div2(A)
    B = east(A)
    B = div2(B)
    C = west(B)
    A = add(A, C)
    C = south(A)
    C = west(C)
    A = add(A, C)
    A = add(B, A)
    B = south(B)
    B = sneg(B)
    A = add(B, A)
    B = add(B, B)
    A = add(A, B)
    B = west(B)
    B = sneg(B)
    B = south(B)
    A = add(A, B)
    B = east(B)
    B = div2(B)
    B = sneg(B)
    C = north(B)
    C = north(C)
    B = sub(B, C)
    A = add(A, B)
    B = west(B)
    B = west(B)
    A = add(A, B)
    B = east(B)
    B = east(B)
    A = add(A, B)

    return A
def conv1_kernel2(input):
    A=input

    A = north(A)
    A = div2(A)
    A = div2(A)
    B = east(A)
    B = div2(B)
    A = add(A, B)
    B = south(B)
    B = south(B)
    B = sneg(B)
    A = add(A, B)
    B = west(B)
    B = west(B)
    A = add(A, B)

    return A
def conv1_kernel3(input):
    A=input

    A = div2(A)
    A = div2(A)
    A = div2(A)
    A = sneg(A)
    B = north(A)
    B = east(B)
    A = add(A, B)
    A = add(A, B)
    A = add(A, B)
    B = south(B)
    B = south(B)
    B = sneg(B)
    A = add(B, A)
    B = west(B)
    B = sneg(B)
    A = add(A, B)
    B = west(B)
    A = add(A, B)
    B = add(B, B)
    A = add(A, B)
    B = east(B)
    A = add(A, B)
    B = north(B)
    A = add(A, B)
    B = north(B)
    A = add(A, B)
    B = south(B)
    B = east(B)
    B = div2(B)
    A = add(A, B)

    return A

def conv2_kernel1(input):
    A=input

    A = div2(A)
    A = div2(A)
    A = sneg(A)
    B = south(A)
    A = north(A)
    A = add(A, B)
    B = east(B)
    B = div2(B)
    B = sneg(B)
    A = add(A, B)
    B = west(B)
    B = west(B)
    B = sneg(B)
    A = add(A, B)
    A = add(B, A)
    A = add(A, B)
    B = north(B)
    C = north(B)
    B = add(B, C)
    A = add(A, B)
    B = east(B)
    B = east(B)
    A = add(A, B)

    return A
def conv2_kernel2(input):
    A=input

    A = div2(A)
    A = div2(A)
    A = div2(A)
    B = north(A)
    B = west(B)
    B = sneg(B)
    C = south(B)
    A = east(A)
    A = add(A, C)
    C = south(A)
    A = sub(A, C)
    A = add(B, A)
    A = add(B, A)
    B = east(B)
    C = east(B)
    B = add(B, C)
    A = add(A, B)
    B = south(B)
    B = south(B)
    A = add(A, B)

    return A
def conv2_kernel3(input):
    A=input

    A = south(A)
    A = div2(A)
    A = div2(A)
    A = div2(A)
    B = east(A)
    B = sneg(B)
    C = north(B)
    C = north(C)
    B = add(B, B)
    B = add(B, C)
    B = add(A, B)
    A = west(A)
    A = sneg(A)
    B = add(B, A)
    B = add(B, A)
    B = add(B, A)
    A = north(A)
    A = sneg(A)
    C = north(A)
    A = add(A, C)
    B = add(B, A)
    A = east(A)
    A = sneg(A)
    B = add(B, A)
    A = east(A)
    A = add(B, A)

    return A

def conv2_kernel1_col(input):
    A=input

    A = div2(A)
    A = div2(A)
    A = sneg(A)
    B = east(A)
    A = west(A)
    A = add(A, B)
    B = north(B)
    A = add(A, B)
    B = div2(B)
    A = add(A, B)
    B = south(B)
    B = south(B)
    B = sneg(B)
    A = add(B, A)
    B = west(B)
    B = sneg(B)
    C = north(B)
    C = north(C)
    B = add(B, C)
    A = add(A, B)
    B = west(B)
    A = add(A, B)

    return A
def conv2_kernel2_col(input):
    A= input

    A = div2(A)
    A = div2(A)
    A = div2(A)
    B = north(A)
    B = west(B)
    B = sneg(B)
    C = east(B)
    A = south(A)
    A = add(A, C)
    C = east(A)
    A = sub(A, C)
    A = add(B, A)
    A = add(B, A)
    B = south(B)
    C = east(B)
    C = east(C)
    B = add(B, C)
    A = add(A, B)
    B = south(B)
    A = add(A, B)

    return A
def conv2_kernel3_col(input):

    A=input

    A = north(A)
    A = east(A)
    A = div2(A)
    A = div2(A)
    A = sneg(A)
    B = div2(A)
    A = add(A, B)
    B = south(B)
    B = sneg(B)
    A = add(A, B)
    B = south(B)
    B = sneg(B)
    A = add(A, B)
    A = add(B, A)
    B = west(B)
    B = west(B)
    A = add(A, B)
    B = north(B)
    B = north(B)
    B = sneg(B)
    C = east(B)
    B = add(B, C)
    A = add(A, B)
    B = south(B)
    B = sneg(B)
    A = add(B, A)
    B = south(B)
    A = add(A, B)

    return A


model=SCAMP_CNNmodel.CNN(256,256,1,2)
model2=SCAMP_CNNmodel.CNN2(64,64,1,2)
model.load_weights('6kernels.h5')

model2.layers[3].set_weights(model.layers[7].get_weights())
model2.layers[4].set_weights(model.layers[8].get_weights())
model2.layers[5].set_weights(model.layers[9].get_weights())
print("done")

graph = tf.get_default_graph()

def image_callback(msg):

    global model
    global turn
    global linear
    global l1
    global l2
    global l3
    global l4
    global count, maxturn

    cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
    cv2_gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

    cv2_res = cv2.resize(cv2_gray, dsize=(256, 256))  # Resize
    im = np.asarray(cv2_res)

    A=list(im)
    B=list(im)
    C=list(im)

    #=================conv1_kernel1===============
    A=conv1_kernel1(A)
    A=stride2(A)
    A=np.array(A)
    for i in range(128):
        for j in range(128):
            A[i][j]=A[i][j]-0.06296267
        #A=A.tolist()
        #A=relu(A)
        #A=np.array(A)


        #=================conv1_kernel2===============
    B=conv1_kernel2(B)
    B=stride2(B)
    B=np.array(B)
    for i in range(128):
        for j in range(128):
            B[i][j]=B[i][j]-0.20187585
        #B=B.tolist()
        #B=relu(B)
        #B=np.array(B)


        #=================conv1_kernel3===============
    C=conv1_kernel3(C)
    C=stride2(C)
    C=np.array(C)
    for i in range(128):
        for j in range(128):
            C[i][j]=C[i][j]-0.03857995
        #C=C.tolist()
        #C=relu(C)
        #C=np.array(C)


        #=================conv2==================

    A=conv2_kernel1(A)
    A=stride2(A)
    A=np.array(A)

    B=conv2_kernel2(B)
    B=stride2(B)
    B=np.array(B)

    C=conv2_kernel3(C)
    C=stride2(C)
    C=np.array(C)

    for i in range(64):
        for j in range(64):
            A[i][j]=A[i][j]+B[i][j]+C[i][j]+0.30638465

    A=A.tolist()
    A=relu(A)
    A=np.array(A)

    A=highlight(A)                                                                      #  Hightlight 

    A=A.reshape([1, 64, 64, 1])

    with graph.as_default():
        result = model2.predict(A)
        turnsignal = result[0]
        turnsignal = turnsignal[0]
        straightsignal = result[1]
        straightsignal = straightsignal[0]

    l1.append(turnsignal[0])
    l2.append(straightsignal[0])
    l3.append(turnsignal[1])
    l4.append(straightsignal[1])
    count = count+1
    if count==5:

	avgL=sum(l1)/5
	avgS=sum(l2)/5
	avgR=sum(l3)/5
	avgB=sum(l4)/5


        linear = (1-0.7)*linear + 0.7*(avgS-avgB)*0.6  # was 1.5
        turn = (1-0.7)*turn + 0.7*(avgR-avgL)*1

        print("LinearV: {} AngularV: {}".format(linear, turn))
	print("AvgL: {} AvgS: {} AvgR: {}, avgB: {}".format(avgL, avgS, avgR, avgB))
	print("====================")
	mov = movement()
	mov.move_command()

        count=0
	l1=[]
	l2=[]
	l3=[]
	l4=[]



#=================maxpooling, dense, sofmax==================
#A=A.reshape([1,64,64,1])
#result = model2.predict(A)
#print(result[0])
#print(result[1])


#command===============================================
class movement :

    def __init__(self):
        self.pub_move = rospy.Publisher("/cmd_vel",Twist,queue_size=10)
	self.move=Twist()

    def move_command(self):

	self.move.angular.z=turn
	self.move.linear.x=linear
	self.pub_move.publish(self.move)

	#print(self.move)

#========================================Main==============================================
rospy.init_node("Sim")
image_sub=rospy.Subscriber("/camera/rgb/image_raw", Image,image_callback)
rospy.spin()

