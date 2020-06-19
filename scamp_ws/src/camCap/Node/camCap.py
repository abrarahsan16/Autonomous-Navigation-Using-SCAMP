#!/usr/bin/env python
#This is an experimental file for the purpose of testing image conversion from cv2 to numpy.
#The images saved in the img folder are for demonstration.

#Importing the dependencies
import rospy
import cv2
import os
import numpy as np
import pandas as pd

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
#Commented out because we do not need tensorflow or keras right now, but these will be implemented in the neural network
#import tensorflow as tf
#from keras.preprocessing import image
path = "/home/abrarahsan16/SCAMP/Autonomous-Navigation-Using-SCAMP/scamp_ws/src/camCap/Node/img/"
bridge = CvBridge()

df=pd.DataFrame({"Time":[0.0],"imageArray":[np.zeros((480,640),dtype=int)]},columns=["Time","imageArray"])#placeholder

def callback(msg):
    global df
    print("Received")
    try:
        cv2_img = bridge.imgmsg_to_cv2(msg,"bgr8") #Capture the image
        cv2_gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY) #convert it to gray
    except CvBridgeError, e:
        print(e)
    else:
        np_img = np.asarray(cv2_gray) #convert to a numpy array
        time = msg.header.stamp #capture the time on the image
	seconds=rospy.get_time()
	
        cv2.imwrite(os.path.join(path,""+str(time)+".jpeg"),np_img)  #save the image. 
	print("Printed")
	
	df1=pd.DataFrame({"Time":[seconds],"imageArray":[np_img]},columns=["Time","imageArray"])
	df=df.append(df1,ignore_index=True)
	df.to_csv('/home/andrew/turtlebot3_ws/imageArray.csv') #NOTE please change the saving path

def main():
    #Start ROS node
    rospy.init_node('camCap')
    rospy.Subscriber("/camera/rgb/image_raw", Image, callback)
    rospy.spin()

if __name__ == '__main__':
    main()
