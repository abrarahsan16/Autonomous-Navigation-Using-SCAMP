#!/usr/bin/env python

#Importing the dependencies
import rospy
import cv2
import os
import numpy as np
import pandas as pd

from sensor_msgs.msg import Imu
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

path = "/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/data_recorder/data/" #Change according to need
bridge = CvBridge()

df = pd.DataFrame({"Time":[0.0],"LinearV":[0.0], "AngularV":[0.0]},columns=["Time","LinearV","AngularV"])

def callback(msg):
    global time
    try:
        cv2_img = bridge.imgmsg_to_cv2(msg,"bgr8") #Capture the image
        cv2_gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY) #convert it to gray
    except CvBridgeError, e:
        print(e)
    else:
        np_img = np.asarray(cv2_gray) #convert to a numpy array
        time = msg.header.stamp #capture the time on the image
        cv2.imwrite(os.path.join(path,"img/"+str(time)+".jpeg"),np_img)  #save the image.


def imu_callback(msg):
    global df
    LinearV = round(msg.linear_acceleration.x,2)
    AngularVel = round(msg.angular_velocity.z,3)
    df1 = pd.DataFrame({"Time":[time],"LinearV":[LinearV],"AngularV":[AngularVel]},columns=["Time","LinearV","AngularV"])
    df = df.append(df1,ignore_index=True)
    df.to_csv(os.path.join(path,'Velocity.csv'))# NOTE: please change the saving path
    print(str(LinearV) + " " + str(AngularVel))



rospy.init_node("recorder")
rospy.Subscriber("/camera/rgb/image_raw", Image, callback)
rospy.Subscriber("/imu",Imu,imu_callback)

rate=rospy.Rate(1)

while not rospy.is_shutdown():
	rate.sleep()
