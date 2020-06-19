#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Imu
import pandas as pd

df=pd.DataFrame({"Time":[0.0],"AngularV":[0.0]},columns=["Time","AngularV"])

def callback(msg):

	global df
	
	AngularVel=round(msg.angular_velocity.z,3)
	print("Angular velocity is: ",AngularVel)
	
	time=msg.header.stamp
	print("    at time: ")
	seconds=rospy.get_time()
	print(seconds)

	df1=pd.DataFrame({"Time":[seconds],"AngularV":[AngularVel]},columns=["Time","AngularV"])
	df=df.append(df1,ignore_index=True)
	df.to_csv('/home/andrew/turtlebot3_ws/AngularVelocity.csv')# NOTE: please change the saving path  


rospy.init_node("AngularVelocity")
sub=rospy.Subscriber("/imu",Imu,callback)
		
rate=rospy.Rate(1)

while not rospy.is_shutdown():
	rate.sleep() 






