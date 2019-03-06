

from plotter import Plotter
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt


filenames = [file for file in listdir('./logs/') if isfile(join('./logs/', file))]

# Plot everything.
indexes=range(len(filenames))
for file in filenames:
	if file[:17]=='double_pendulum_P':
		try:
			plotter = Plotter("./logs/"+file)
			plotter.plot_scalar_fields(["mean_return"])
			plt.title('PPO')
			plotter.show()
		except EOFError:
			print('PASSED FILE: '+file)
			pass
	elif file[:17]=='double_pendulum_R':
		try:
			print(file[16:])
			plotter = Plotter("./logs/"+file)
			plotter.plot_scalar_fields(["mean_return"])
			plt.title('Reinforce')
			plotter.show()
		except EOFError:
			print('PASSED FILE: '+file)
			pass
