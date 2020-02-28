#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt

################################
#
# This script is meant to read the "progress.txt" file output by ppo_node.py 
# and plot the learning curve for the training data
# This file will be located in the folder you set in ppo_node.py
# If this script is not in the same folder as "progress.txt" put the path in PREFIX
#
################################



PREFIX = ""

average_error = []

with open(PREFIX + 'progress.txt', 'r') as file:
    for line in file:
        words = line.split()
        average_error.append(words[1])

nums = [float(i) for i in average_error[1:]]

print len(nums)
print nums

plt.plot(np.array(nums))
# plt.plot(r)
plt.show()



