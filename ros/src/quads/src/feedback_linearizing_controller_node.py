#!/usr/bin/python

from feedback_linearizing_controller import FeedbackLinearizingController

import rospy
import sys

if __name__ == "__main__":
    rospy.init_node("feedback_linearizing_controller")

    ctrl = FeedbackLinearizingController()

    rospy.spin()
