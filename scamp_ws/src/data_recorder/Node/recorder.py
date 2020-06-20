#!/usr/bin/env python

#Importing the dependencies
import rospy
import cv2
import os
import numpy as np
import pandas as pd
import message_filters

from sensor_msgs.msg import Imu
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

#Change according to user
#path_vel = "/home/andrew/turtlebot3_ws/data/"
#path_img = "/home/andrew/turtlebot3_ws/data/images/"
path_vel = "/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/data_recorder/data/"
path_img = "/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/data_recorder/data/img/"

bridge = CvBridge()

df = pd.DataFrame({"Time":[0.0],"LinearA":[0.0], "AngularV":[0.0]},columns=["Time","LinearA","AngularV"])

def mycall(image,imu): # "image" and "imu" are just "msgs" we used before in seperate callback fucntions

#================Define variables================
	global df
	time = image.header.stamp     #  image. and imu. should get same value.

#================Image processing================

	cv2_img = bridge.imgmsg_to_cv2(image,"bgr8") 	#Capture the image
        cv2_gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY) 	#convert it to gray
	np_img = np.asarray(cv2_gray)
	cv2.imwrite(os.path.join(path_img,str(time)+".jpeg"),np_img)

#================Velocity processing================

	LinearA = round(imu.linear_acceleration.x,4)
    	AngularVel = round(imu.angular_velocity.z,4)
	df1 = pd.DataFrame({"Time":[time],"LinearA":[LinearA],"AngularV":[AngularVel]},columns=["Time","LinearA","AngularV"])
    	df = df.append(df1,ignore_index=True)

	df.to_csv(os.path.join(path_vel,'Velocity.csv'))

#================print current time================
	print(time)





rospy.init_node("DataCollection")
image_sub= message_filters.Subscriber("/camera/rgb/image_raw", Image)
imu_sub= message_filters.Subscriber("/imu",Imu)

ts = message_filters.TimeSynchronizer([image_sub,imu_sub], 10)
ts.registerCallback(mycall)
rospy.spin()
