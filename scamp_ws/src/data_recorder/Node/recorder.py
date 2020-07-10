#!/usr/bin/env python

#Importing the dependencies
import rospy
import cv2
import os
import numpy as np
import pandas as pd
import message_filters

from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

#Change according to user
#path_vel = "/home/andrew/turtlebot3_ws/data/"
#path_img = "/home/andrew/turtlebot3_ws/data/images/"
path_vel = "/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/data_recorder/data/"
path_img = "/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/data_recorder/data/img/"

bridge = CvBridge()

df = pd.DataFrame({"Time":[0.0],"LinearV":[0.0], "AngularV":[0.0]},columns=["Time","LinearV","AngularV"])

def mycall(image,vel): # "image" and "imu" are just "msgs" we used before in seperate callback fucntions

#================Define variables================
	global df
	time = image.header.stamp     #  image. and imu. should get same value.

#================Image processing================

	cv2_img = bridge.imgmsg_to_cv2(image,"bgr8") 	#Capture the image
    	cv2_gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY) 	#convert it to gray
	np_img = np.asarray(cv2_gray)
	cv2.imwrite(os.path.join(path_img,str(time)+".jpeg"),np_img)

#================Velocity processing================

	LinearV = round(vel.twist.linear.x,4)
    	AngularVel = round(vel.twist.angular.z,4)
	df1 = pd.DataFrame({"Time":[time],"LinearV":[LinearV],"AngularV":[AngularVel]},columns=["Time","LinearV","AngularV"])
    	df = df.append(df1,ignore_index=True)

	df.to_csv(os.path.join(path_vel,'Velocity.csv'))

#================print current time================
	rospy.loginfo(": [%f, %f]"%(vel.twist.linear.x, vel.twist.angular.z))


rospy.init_node("DataCollection")
image_sub= message_filters.Subscriber("/camera/rgb/image_raw", Image)
vel_sub= message_filters.Subscriber("vel_timer",TwistStamped)

ts = message_filters.TimeSynchronizer([image_sub,vel_sub], 10)
ts.registerCallback(mycall)
rospy.spin()
