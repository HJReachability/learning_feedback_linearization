

from plotter import Plotter
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt


filenames = [file for file in listdir('./logs/') if isfile(join('./logs/', file))]

#plotter = Plotter("./logs/quadrotor_14d_Reinforce_3x10_std0.100000_lr0.001000_kl-1.000000_50_25_dyn_0.900000_1.100000_1.100000_1.100000_seed_228.pkl")
#plotter = Plotter("./logs/quadrotor_14d_Reinforce_3x10_std0.100000_lr0.001000_kl-1.000000_50_25_dyn_0.500000_0.500000_0.500000_0.500000_seed_95.pkl")
#plotter = Plotter("./logs/quadrotor_14d_Reinforce_3x10_std1.000000_lr0.001000_kl-1.000000_100_25_fromzero_False_dyn_1.100000_0.500000_0.500000_0.500000_seed_204_smallweights.pkl") # swweeeet
#plotter = Plotter("./logs/quadrotor_14d_Reinforce_3x10_std1.000000_lr0.000100_kl-1.000000_50_25_fromzero_False_dyn_1.100000_0.500000_0.500000_0.500000_seed_180_smallweights.pkl") #nice!
plotter = Plotter("./logs/quadrotor_14d_Reinforce_3x10_std0.500000_lr0.000100_kl-1.000000_50_25_fromzero_False_dyn_1.100000_0.500000_0.500000_0.500000_seed_440_smallweights.pkl") #nice!
plotter.plot_scalar_fields(["mean_return"])
plt.title('Reinforce')
plotter.show()


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
	elif file[:17]=='quadrotor_14d_R':
		try:
			print(file[16:])
			plotter = Plotter("./logs/"+file)
			plotter.plot_scalar_fields(["mean_return"])
			plt.title('Reinforce')
			plotter.show()
		except EOFError:
			print('PASSED FILE: '+file)
			pass
