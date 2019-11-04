#!/usr/bin/env python


# Python Imports
import sys
import numpy as np
import itertools

import dill
import matplotlib.pyplot as plt

if __name__ == '__main__':
	r = dill.load(open('rewards.pkl', 'rb'))

	epochs = len(r)/1250

	epoch_rewards = []

	for i in range(epochs):
		epoch_rewards.append(np.mean(r[1250*i:1250*(i+1)]))

	plt.plot(np.array(epoch_rewards))
	# plt.plot(r)
	plt.show()





