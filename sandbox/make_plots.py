


from plotter import Plotter

# Plot everything.
filename = "./logs/double_pendulum_4x20_std0.250000_lr0.001000_kl0.100000_100_10_dyn_0.900000_1.500000_1.000000_1.000000_1.000000_seed_102.pkl"
plotter = Plotter(filename)
plotter.plot_scalar_fields(
    ["mean_return"], title="Mean return")
plotter.plot_scalar_fields(
    ["learning_rate"], title="Learning rate")

plotter.show()
