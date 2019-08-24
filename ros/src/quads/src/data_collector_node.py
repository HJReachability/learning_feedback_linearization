#!/usr/bin/python

from data_collector import DataCollector

import rospy
import sys

if __name__ == "__main__":
    rospy.init_node("data_collector")

    collector = DataCollector()

    rospy.spin()
    collector.dump()
