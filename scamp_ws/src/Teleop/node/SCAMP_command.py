#!/usr/bin/env python


import rospy
import numpy as np
import cv2
#import tensorflow as tf
# from Keras import....

#import SCAMP_CNN as SC

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

from cv_bridge import CvBridge, CvBridgeError
bridge = CvBridge()


data=np.array([])


def image_callback(msg):
	
	
	cv2_img = bridge.imgmsg_to_cv2(msg,"bgr8") 
        cv2_gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY) 
        np_img = np.asarray(cv2_gray)
	print(np_img)

	global data
	
	
	#//////////////// trained CNN ///////////////////////
	

# ==> input: variable=np_img
	# to
 	# Convolution and Max Pooling layers....
	#   ...........
	#   ...........
	# should have two FCs, both doing regression
	# FC_LinearVel:  1 input
	# FC_AngularVel: 1 input

	# (if using collision prob, 1 input, 2 outputs(0 or 1)



							# ==> outputs: outputlist=[LinearV,AngularV]
	
	#///////////////////////////////////////////////////


	# it should be data=np.array(outputlist),for demonstration I assume it's [1,-1]
	data=np.array([1,-1])
#---------------------------
	mov = movement()   #
	mov.move_command() #
#---------------------------
	#
       #
      #
     #
class movement :

    def __init__(self):
        self.pub_move = rospy.Publisher("/cmd_vel",Twist)
	self.move=Twist() 
	
    def move_command(self):
	self.move.linear.x=data[0]
	self.move.angular.z=data[1]
	self.pub_move.publish(self.move)

	print(self.move)


rospy.init_node("Command")
image_sub=rospy.Subscriber("/camera/rgb/image_raw", Image,image_callback)
rospy.spin()





