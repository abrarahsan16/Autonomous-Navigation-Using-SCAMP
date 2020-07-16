#!/usr/bin/env python


import rospy
import numpy as np
import cv2
import tensorflow as tf

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()

turn=0

import SCAMP_CNNmodel 


model=SCAMP_CNNmodel.CNN(256,256,1,3)
model.load_weights('my_model_weights.h5') #change according to need
graph = tf.get_default_graph()
print("weight loaded")



#===================================Casting area===================================================
def image_callback(msg):

	global model
	global turn	
	
	cv2_img = bridge.imgmsg_to_cv2(msg,"bgr8") 
        cv2_gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
	cv2_res = cv2.resize(cv2_gray, dsize=(256, 256)) # needs center crop
        np_img = np.asarray(cv2_res)
	np_img=np_img.reshape([1,256,256,1])
	
	with graph.as_default():
     		result= model.predict(np_img)
		
	result=result.flatten()
	print(result)	
	
	index=np.argmax(result, axis=0)
	print("===============")
		
	if index==0:
		print("L")
		turn=-0.4

	if index==1:
		print("S")
		turn=0.0

	if index==2:
		print("R")
		turn=0.4


	mov = movement()   
	mov.move_command() 

#=======================================Set command===============================================
class movement :

    def __init__(self):
        self.pub_move = rospy.Publisher("/cmd_vel",Twist,queue_size=10)
	self.move=Twist() 
	
    def move_command(self):
	
	self.move.angular.z=turn
	self.move.linear.x=0.1
	self.pub_move.publish(self.move)

	#print(self.move)

#========================================Main==============================================
rospy.init_node("Command")
image_sub=rospy.Subscriber("/camera/rgb/image_raw", Image,image_callback)
rospy.spin()
