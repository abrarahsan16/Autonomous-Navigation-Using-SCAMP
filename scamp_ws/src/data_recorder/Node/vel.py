#!/usr/bin/env python

import rospy
import os
import numpy as np
import message_filters

from geometry_msgs.msg import Twist, TwistStamped
import time

def velCallback(msg):
    lin = msg.twist.linear.x
    steer = msg.twist.angular.z
    rospy.loginfo("Components: [%f, %f]"%(lin, steer))

twist = TwistStamped()
rospy.init_node("VelocityCollection")
rospy.Subscriber("vel_timer", TwistStamped, velCallback)
rospy.spin()
