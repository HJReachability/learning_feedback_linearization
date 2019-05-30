

from plotter import Plotter
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt


filenames = [file for file in listdir('./logs/') if isfile(join('./logs/', file))]

#plotter = Plotter("./logs/quadrotor_14d_Reinforce_3x10_std0.500000_lr0.000100_kl-1.000000_50_25_fromzero_False_dyn_1.100000_0.500000_0.500000_0.500000_seed_440_smallweights.pkl") #nice!
#plotter.plot_scalar_fields(["mean_return"])
#plt.title('Reinforce')
#plotter.show()


# Plot everything.
indexes=range(len(filenames))
for file in filenames:
#    if file[:15]=='quadrotor_14d_R' and file[-2:] == '_6':
#        try:
#            print(file[16:])
#            plotter = Plotter("./logs/"+file)
#            plotter.plot_scalar_fields(["mean_return"])
#            plt.title('Reinforce')
#            plotter.show()
#        except EOFError:
#            print('PASSED FILE: '+file)
#            pass
#    if file[:15]=='quadrotor_12d_R' and file[-2:] == '_6':
#        try:
#            print(file[16:])
#            plotter = Plotter("./logs/"+file)
#            plotter.plot_scalar_fields(["mean_return"])
#            plt.title('Reinforce')
##            plotter.plot_scalar_fields(["stddev"])
#            plotter.show()
#        except EOFError:
#            print('PASSED FILE: '+file)
#            pass
    if file[:17]=='double_pendulum_P' and file[-2:] == '_6':
        try:
            plotter = Plotter("./logs/"+file)
            plotter.plot_scalar_fields(["mean_return"])
            plt.title('PPO')
            plotter.show()
        except EOFError:
            print('PASSED FILE: '+file)
            pass
    elif file[:17]=='double_pendulum_R' and file[-2:] == '_2':
        try:
            print(file[16:])
            plotter = Plotter("./logs/"+file)
            plotter.plot_scalar_fields(["mean_return"])
            plt.title('Reinforce')
            plotter.show()
        except EOFError:
            print('PASSED FILE: '+file)
            pass
