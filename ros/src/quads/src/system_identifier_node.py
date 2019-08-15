#!/usr/bin/python

from system_identifier import SystemIdentifier

import rospy
import sys



if __name__ == "__main__":
    rospy.init_node("system_identifier")

    sysid = SystemIdentifier()
    rate = rospy.Rate(1.0)
    while not rospy.is_shutdown():
        rate.sleep()
        sysid.solve()
