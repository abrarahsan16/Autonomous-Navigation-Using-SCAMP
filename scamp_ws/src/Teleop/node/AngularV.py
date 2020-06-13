#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Imu

def callback(msg):

	print("Angular velocity is: ",round(msg.angular_velocity.z,3))
	time=msg.header.stamp
	print("    at time: ")
	print(time)

rospy.init_node("AngularVelocity")
sub=rospy.Subscriber("/imu",Imu,callback)
		
rate=rospy.Rate(1)

while not rospy.is_shutdown():
	
	rate.sleep() 






