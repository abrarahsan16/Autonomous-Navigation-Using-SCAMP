#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import numpy as np


data=np.array([])

class movement :

    def __init__(self):
        rospy.init_node('move_robot_node', anonymous=False)
        self.pub_move = rospy.Publisher("/cmd_vel",Twist,queue_size=10)
	self.sub_array= rospy.Subscriber("/camera/rgb/image_raw", Image,queue_size=10)
	# sub_array does not do anything for now, just to show how it could be combined with camCap node

	self.move=Twist() 


    def publish_vel(self):

	self.pub_move.publish(self.move)
	# The step where Twist msgs is actually published to /cmd_vel
	
	print(self.move)
	# show the Twist msgs we're going to publish to /cmd_vel

    def move_command(self): # assume the input of this node is a 2x3 array
	self.move.linear.x=data[0][0]
	self.move.linear.y=data[0][1]
	self.move.linear.z=data[0][2]
        self.move.angular.x=data[1][0]
	self.move.angular.y=data[1][1]
	self.move.angular.z=data[1][2]


if __name__=="__main__":

	mov = movement()
        rate=rospy.Rate(1)

	while not rospy.is_shutdown():

		print("please enter direction array:\n")
		data=np.array(input()) # for testing, enter list
		mov.move_command()

		mov.publish_vel()
		rate.sleep() 
    
