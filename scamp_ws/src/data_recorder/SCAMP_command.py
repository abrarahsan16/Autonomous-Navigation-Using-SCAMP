#!/usr/bin/env python

#import vis
import rospy
import numpy as np
import cv2
import tensorflow as tf
import statistics as st

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from keras.models import Model
from matplotlib import pyplot as plt
from matplotlib.pyplot import draw


import keras
#import vis
#from vis.utils import utils
#from vis.visualization import visualize_cam


bridge = CvBridge()

turn = 0
linear = 0
count=0
l1=[]
l2=[]
l3=[]


import SCAMP_CNNmodel


model=SCAMP_CNNmodel.CNN(256,256,1,3)
model.load_weights('my_model_weights.h5') #change according to need
graph = tf.get_default_graph()
print("weight loaded")

#===================================Casting area===================================================
def image_callback(msg):

	global model
	global turn
	global linear
	global l1
	global l2
	global l3
	global count

	cv2_img = bridge.imgmsg_to_cv2(msg,"bgr8")
        cv2_gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
	cv2_res = cv2.resize(cv2_gray, dsize=(256, 256)) # needs center crop
        np_img = np.asarray(cv2_res)
	np_img=np_img.reshape([1,256,256,1])

	with graph.as_default():
     		result= model.predict(np_img)

		#Note the code must be under "with graph.as_default():" for live data stream,
		#otherwise a tensor error will raise

		mapmodel=Model(inputs=model.inputs,outputs=model.layers[2].output)
		# 3--> layer of "max_pooling2d_2 (MaxPooling2)"
		# change to 7 for heat map

#=============================Heat Map Generation==================================================

#		layer_idx=utils.find_layer_idx(mapmodel, 'activation_1')
#		mapmodel.layers[layer_idx].activation = keras.activations.linear
#		model2= utils.apply_modifications(mapmodel)

#		y_pred=model2.predict(np_img)
#		class_idxs_sorted =np.argsort(y_pred.flatten())[::-1]
#		test=np.argsort(y_pred.flatten())

#		penultimate_layer_idx = utils.find_layer_idx(model2, "conv2d_1_input")
#		class_idx=class_idxs_sorted[0]

#		seed_input=np_img

#		grad_top1= visualize_cam(model2, layer_idx,[0],seed_input,
#                           penultimate_layer_idx = penultimate_layer_idx,
#                           backprop_modifier= None,
#                           grad_modifier= None)


#		fig,axes=plt.subplots(1,2,figsize=(14,5))
#		axes[0].imshow(cv2_img)
#		axes[1].imshow(cv2_img)
#		i=axes[1].imshow(grad_top1,cmap="jet",alpha=0.3)
#		fig.colorbar(i)
#		pyplot.savefig("/home/andrew/Z_feature_map/map")
#		plt.close()

#=======================================Feature Map=================================================

		#print("==================")
		#print(Layer.summary())
		#print("==================")


		#fm=mapmodel.predict(np_img)
		#plt.figure()
		#plt.show(block=False)
		#plt.ion()
		#plt.imshow(fm[0,:,:,2], cmap='viridis')
		#2--> the feature from the 3rd fillter
		#pyplot.savefig("/home/andrew/Z_feature_map/feature")
		#plt.pause(1) #change accordingly
		#plt.savefig("/home/abrarahsan16/feature_map/feature")
		#plt.show()
		#plt.close()

#===========================================================================================

	result=result.flatten()
	#print(result)

#===========================================================================================

	l1.append(result[0])
	l2.append(result[1])
	l3.append(result[2])
	count = count+1
	if count==3:

		avgL=sum(l1)/3
		avgS=sum(l2)/3
		avgR=sum(l3)/3


		#avgL = st.median(l1)
		#avgS = st.median(l2)
		#avgR = st.median(l3)

		#print("{} {} {}".format(avgL, avgS, avgR))


		print("LinearV: {} AngularV: {}".format(linear, turn))

		if avgS>0.95:
			print("S")
			linear = 0.1
			turn = (avgR - avgL)*0.4
		else:
			linear = 0
			if avgS<0.8:
				print("B")
				linear = -0.01
			else:
				if avgL>avgR:
					print("L")
					turn = 0.4

				if avgL<avgR:
					print("R")
					turn = -0.4

		#print("l1: "+str(l1))
		#print("l2: "+str(l2))
		#print("l3: "+str(l3))

		#print("AvgL: "+str(avgL))
		#print("AvgS: "+str(avgS))
		#print("AvgR: "+str(avgR))
		print("AvgL: {} AvgS: {} AvgR: {}".format(avgL, avgS, avgR))
		print("====================")
		mov = movement()
		mov.move_command()

		count=0
		l1=[]
		l2=[]
		l3=[]
#===========================================================================================

#	index=np.argmax(result, axis=0)

#	print("{} {}".format(linear, turn))
#	linear = result[1] * 0.1
	#turn = 0.4*(result[0] - result[2])

#	if result[0] > result[2]:
#		print("L")
		#turn=0.4
		#linear=0
#		turn = 0.4

	#if result[1] > 0:
		#print("S")
		#turn=0.0
		#linear=0.05

#	if result[0] < result[2]:
#		print("R")
		#turn=-0.4
		#linear=0
#		turn = -0.4

#	if result[0] == 0 and result[2] == 0:
#		turn=0

	#if result[0] == 0 and result[1] == 0 and result[2] == 0:
		#linear = -0.02
		#turn = 0.3

#	print("===============")
#	mov = movement()
#	mov.move_command()

#=======================================Set command===============================================
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
rospy.init_node("Command")
image_sub=rospy.Subscriber("/camera/rgb/image_raw", Image,image_callback)
rospy.spin()
