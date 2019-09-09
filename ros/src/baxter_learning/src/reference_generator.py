#!/usr/bin/env python

# Python Imports
import sys
import numpy as np

# ROS imports
import rospy
from baxter_learning_msgs.msg import State, Reference
from std_msgs.msg import Empty

class ReferenceGenerator(object):
    """docstring for ReferenceGenerator"""
    def __init__(self):
        
        if not self.load_parameters(): sys.exit(1)

        rospy.on_shutdown(self.shutdown)

        if not self.register_callbacks(): sys.exit(1)


    def load_parameters(self):

        if not rospy.has_param("~topics/ref"):
            return False
        self._ref_topic = rospy.get_param("~topics/ref")

        if not rospy.has_param("~topics/linear_system_reset"):
            return False
        self._reset_topic = rospy.get_param("~topics/linear_system_reset")

        return True

    def register_callbacks(self):

        self._reset_sub = rospy.Subscriber(
            self._reset_topic, Empty, self.linear_system_reset_callback)

        self._ref_pub = rospy.Publisher(
            self._ref_topic, Reference)

        return True

    def send_zeros(self):
        setpoint = State(np.zeros(7), np.zeros(7))
        feed_forward = np.zeros(7)
        msg = Reference(setpoint, feed_forward)
        while not rospy.is_shutdown():
            self._ref_pub.publish(msg)
            rospy.sleep(0.05)

    def linear_system_reset_callback(self, msg):
        rospy.sleep(0.5)

    def shutdown(self):
        pass

if __name__ == '__main__':
    
    rospy.init_node("reference_generator")

    gen = ReferenceGenerator()

    gen.send_zeros()

    rospy.spin()






        





