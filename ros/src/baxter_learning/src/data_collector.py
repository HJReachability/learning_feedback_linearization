#!/usr/bin/env python

import rospy
import numpy as np

from baxter_learning_msgs.msg import State, DataLog
import sys

class DataCollector(object):
    def __init__(self):
        
        if not self.load_parameters(): sys.exit(1)
        self._refs = []
        self._states = []
        self._times = []

        rospy.on_shutdown(self.shutdown)

        if not self.register_callbacks(): sys.exit(1)


    def load_parameters(self):
        if not rospy.has_param("~topics/data"):
            return False
        self._data_topic = rospy.get_param("~topics/data")

        return True

    def register_callbacks(self):
        self._data_sub = rospy.Subscriber(
            self._data_topic, DataLog, self.callback)

        return True

    def callback(self, msg):
        t = rospy.Time.now().to_sec()
        r = np.hstack([msg.ref.position, msg.ref.velocity])
        s = np.hstack([msg.state.position, msg.state.velocity])

        self._refs.append(r)
        self._states.append(s)
        self._times.append(t)

    def dump(self):
        """ Dump to disk. """
        import dill
        PREFIX = "/home/ee106a-tah/Desktop/data"

        f = open(PREFIX + "refs.pkl", "wb")
        dill.dump(self._refs, f)
        f.close()
        f = open(PREFIX + "states.pkl", "wb")
        dill.dump(self._states, f)
        f.close()
        f = open(PREFIX + "times.pkl", "wb")
        dill.dump(self._times, f)
        f.close()

    def shutdown(self):
        rospy.sleep(0.1)
        self.dump()
        rospy.sleep(0.1)

if __name__ == '__main__':
    
    rospy.init_node("data_collector")

    bl = DataCollector()

    rospy.spin()



