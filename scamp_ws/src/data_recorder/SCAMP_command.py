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
maxturn=""
l1=[]
l2=[]
l3=[]
l4=[]


import SCAMP_CNNmodel


model=SCAMP_CNNmodel.CNN(256,256,1,2)
model.load_weights('6kernels.h5') #change according to need

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
	global l4
	global count,maxturn

	cv2_img = bridge.imgmsg_to_cv2(msg,"bgr8")
        cv2_gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

	#========== corridor =======(always work with resize+positive sign)

	#cv2_res = cv2.resize(cv2_gray, dsize=(256, 256)) # needs center crop
	
	#========== track ==========(always work with crop+ negative sign)

	center_width = int(cv2_gray.shape[1]/2)
	center_height = int(450)

	cv2_res = cv2_gray[center_height - int(256):center_height,
						center_width - int(256/2):center_width + int(256/2)]
	
	#===========================#===========================
	np_img=cv2_res.reshape([1,256,256,1])
	np_img = np.asarray(np_img)

	
        #np_img = np.asarray(cv2_res,dtype=np.float32)*np.float32(1.0/255.0)
	#np_img=np_img.reshape([1,256,256,1])

	with graph.as_default():
     		result= model.predict(np_img)
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

		#==========new weight======
		#linear = (1-0.7)*linear + 0.7*(avgS-avgB)*1
		#turn = (1-0.3)*turn + 0.3*(avgR-avgL)*1
	
		

		#linear = (1-0.7)*linear + 0.7*(avgS-avgB)*0.8
		#turn = (1-0.4)*turn + 0.4*(avgR-avgL)*0.8
		
		#if linear<0.3:
		#	linear=0
		#	turn=0.4*(avgR-avgL)*-1.2



		#==========6kernels track ========== 
		linear = (1-0.7)*linear + 0.7*(avgS-avgB)*1.5
		turn = (1-0.7)*turn + 0.7*(avgR-avgL)*1
		
		#========== texture pentagon ==========
		#linear =(1-0.7)*linear + 0.7*(avgS-avgB)*0.8
		#turn = (1-0.7)*turn + 0.7*(avgR-avgL)*1.5

		#========== circletrack ==========
		#linear = (1-0.7)*linear + 0.7*(avgS-avgB)*1
		#turn = (1-0.7)*turn + 0.7*(avgR-avgL)*-1


		#========== corridor ==========
		#linear = (1-0.7)*linear + 0.7*(avgS-avgB)*0.5
		#turn = (1-0.3)*turn + 0.3*(avgR-avgL)*0.6
		
		#linear = (1-0.7)*linear + 0.7*(avgS-avgB)*0.6   #<===
		#turn = (1-0.45)*turn + 0.45*(avgR-avgL)*0.8
		
		#linear = (1-0.7)*linear + 0.7*(avgS-avgB)*0.8
		#turn = (1-0.4)*turn + 0.4*(avgR-avgL)*0.8
		
		#----------------------safe above----------------------------
		#linear = (1-0.7)*linear + 0.7*(avgS-avgB)*0.8
		#turn = (1-0.3)*turn + 0.3*(avgR-avgL)*0.9

		#linear = (1-0.7)*linear + 0.7*(avgS-avgB)*1
		#turn = (1-0.4)*turn + 0.4*(avgR-avgL)*1.2
		
		#========== pentagon ==========
		#linear = (1-0.7)*linear + 0.7*(avgS-avgB)*1
		#turn = (1-0.7)*turn + 0.7*(avgR-avgL)*-1



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
#===========================================================================================

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
rospy.init_node("Command")
image_sub=rospy.Subscriber("/camera/rgb/image_raw", Image,image_callback)
rospy.spin()
