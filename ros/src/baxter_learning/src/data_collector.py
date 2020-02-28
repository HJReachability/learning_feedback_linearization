#!/usr/bin/env python

import rospy
import numpy as np
import time
import datetime
import dill

from std_msgs.msg import String
from baxter_learning_msgs.msg import State, DataLog
from quads_msgs.msg import LearnedParameters, Parameters
from std_srvs.srv import Empty, EmptyResponse
import sys
import os

class DataCollector(object):
    def __init__(self):
        
        if not self.load_parameters(): sys.exit(1)
        self._refs = []
        self._states = []
        self._times = []
        self._outputs = []
        self._rewards = []
        self._learned_parameters = []
        now = datetime.datetime.now()
        self.dt_string = now.strftime("%d-%m-%Y--%H:%M:%S")
        self.notif_pub = rospy.Publisher('data_dump_notif', String, queue_size=10)
        self.save_every = 100

        rospy.on_shutdown(self.shutdown)

        if not self.register_callbacks(): sys.exit(1)


    def load_parameters(self):
        if not rospy.has_param("~topics/data"):
            return False
        self._data_topic = rospy.get_param("~topics/data")

        if not rospy.has_param("~topics/params"):
            return False
        self._param_topic = rospy.get_param("~topics/params")

        if not rospy.has_param("~test_set"):
            return False
        self._is_test_set = rospy.get_param("~test_set")

        return True

    def register_callbacks(self):
        self._data_sub = rospy.Subscriber(
            self._data_topic, DataLog, self.data_callback)

        self._param_sub = rospy.Subscriber(
            self._param_topic, LearnedParameters, self.param_callback)

        self._dump_service = rospy.Service('dump', Empty, self.dump_srv)

        return True

    def data_callback(self, msg):
        t = rospy.Time.now().to_sec()
        ref = np.hstack([msg.ref.position, msg.ref.velocity])
        s = np.hstack([msg.state.position, msg.state.velocity])
        o = msg.transition.a
        rew = msg.transition.r

        self._refs.append(ref)
        self._states.append(s)
        self._times.append(t)
        self._outputs.append(o)
        self._rewards.append(rew)

        if len(self._refs) and len(self._refs) % self.save_every == 0:
            t0 = time.time()
            self.dump()
            t1 = time.time()
            now = datetime.datetime.now()
            self.notif_pub.publish(String(("Saved Data at "
                + now.strftime("%d-%m-%Y--%H:%M:%S")
                + ". Number of timesteps saved: "
                + str(len(self._refs)) + "."
                + " Dump took " + str(t1 - t0) + " seconds.")))

    def param_callback(self, msg):
        p_list = []
        t = rospy.Time.now().to_sec()
        for p in msg.params:
            p_list.append(np.array(p.params))

        self._learned_parameters.append((t, p_list))

    def dump_srv(self, msg):
        self.dump()

        return EmptyResponse()

    def dump(self):
        """ Dump to disk. """
        # Put ABSOLUTE path to logging directory here, then remove the NotImplementedError.
        # eg. PREFIX = "/home/cc/ee106a/fa19/staff/ee106a-taf/Desktop/data"
        PREFIX = "/home/cc/ee106b/sp20/staff/ee106b-laa/Desktop/data"
        # raise NotImplementedError

        filename = PREFIX + "/data_log-" + self.dt_string

        if self._is_test_set:
            np.savez_compressed(filename, refs=self._refs,
                                          states=self._states,
                                          times=self._times,
                                          outputs=self._outputs,
                                          rewards=self._rewards,
                                          learned_params=self._learned_parameters)

        # f = open(PREFIX + "refs.pkl", "wb")
        # dill.dump(self._refs, f)
        # f.close()
        # f = open(PREFIX + "states.pkl", "wb")
        # dill.dump(self._states, f)
        # f.close()
        # f = open(PREFIX + "times.pkl", "wb")
        # dill.dump(self._times, f)
        # f.close()
        # f = open(PREFIX + "outputs.pkl", "wb")
        # dill.dump(self._outputs, f)
        # f.close()
        # f = open(PREFIX + "rewards.pkl", "wb")
        # dill.dump(self._rewards, f)
        # f.close()
        f = open(PREFIX + "/learned_params.pkl", "wb")
        dill.dump(self._learned_parameters, f)
        f.close()

    def shutdown(self):
        # rospy.sleep(0.1)
        # self.dump()
        # rospy.sleep(0.1)

        pass

if __name__ == '__main__':
    
    rospy.init_node("data_collector")

    bl = DataCollector()

    rospy.spin()



