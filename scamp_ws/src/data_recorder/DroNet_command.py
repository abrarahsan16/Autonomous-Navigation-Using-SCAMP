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
from keras.models import model_from_json

import keras
#import vis
#from vis.utils import utils
#from vis.visualization import visualize_cam


bridge = CvBridge()

import DroNet

turn=0
linear=0

json_file = open('model_struct.json', 'r')
loaded_model_json =json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("best_weights.h5")
graph = tf.get_default_graph()
print(loaded_model.summary())

#===================================Casting area===================================================
def image_callback(msg):

	global turn,linear
	cv2_img = bridge.imgmsg_to_cv2(msg,"bgr8")
        cv2_gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
	cv2_res = cv2.resize(cv2_gray, dsize=(200, 200)) 
	
	np_img = np.asarray(cv2_res)
	np_img=np_img.reshape([1,200,200,1])
	with graph.as_default():
     		result=loaded_model.predict(np_img)

	linear=0.05
	steer=result[0][0][0]
	
	# -2 to 7, len=9
	# -1 to 1, len=2	
	subSteer=(steer-(-2))/9*2+(-1)
	
	print("output:  "+str(steer)+"  casted output:  "+str(subSteer))

	turn=0.5*turn+0.5*1.5708*subSteer # formula (3),p4, DroNet: Learning to Fly by Driving
	
	mov=movement()
	mov.move_command()

	
class movement :

    def __init__(self):
        self.pub_move = rospy.Publisher("/cmd_vel",Twist,queue_size=10)
	self.move=Twist()


    def move_command(self):

	self.move.angular.z=turn
	self.move.linear.x=linear
	self.pub_move.publish(self.move)

	#print(self.move)


rospy.init_node("Command")
image_sub=rospy.Subscriber("/camera/rgb/image_raw", Image,image_callback)
rospy.spin()
