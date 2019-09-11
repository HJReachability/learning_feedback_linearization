#!/usr/bin/python

import numpy as np
import dill

PREFIX = "/home/hysys/Github/learning_feedback_linearization/ros/src/quads/data/"

f = open(PREFIX + "refs.pkl", "rb")
refs = dill.load(f)
f.close()

f = open(PREFIX + "ref_times.pkl", "rb")
ref_times = dill.load(f)
f.close()

f = open(PREFIX + "linear_system_states.pkl", "rb")
linear_system_states = dill.load(f)
f.close()

f = open(PREFIX + "linear_system_state_times.pkl", "rb")
linear_system_state_times = dill.load(f)
f.close()

import matplotlib.pyplot as plt

plt.plot(ref_times, [r[0] for r in refs], "b-.")
plt.plot(linear_system_state_times, [x[0] for x in linear_system_states], "r-.")
plt.show()
