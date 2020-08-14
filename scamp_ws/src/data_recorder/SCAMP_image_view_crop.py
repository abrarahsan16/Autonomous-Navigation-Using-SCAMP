#!/usr/bin/env python


import rospy
import numpy as np
import cv2
import os

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


print("started")

bridge = CvBridge()


def image_callback(msg):

	cv2_img = bridge.imgmsg_to_cv2(msg,"bgr8")
	center_width = int(cv2_img.shape[1]/2)
	center_height = int(300)
	cv2_res = cv2_img[center_height - int(256):center_height,
						center_width - int(256/2):center_width + int(256/2)]

        

	image_message = bridge.cv2_to_imgmsg(cv2_res,"bgr8")

	pub = rospy.Publisher("/camera/rgb/crop",Image,queue_size=10)
	pub.publish(image_message)
	

rospy.init_node("image_view_crop")
image_sub=rospy.Subscriber("/camera/rgb/image_raw", Image,image_callback)
rospy.spin()
