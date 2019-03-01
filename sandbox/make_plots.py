


from plotter import Plotter

# Plot everything.
#filename = "./logs/double_pendulum_3x10_std0.050000_lr0.001000_kl0.100000_25_25_dyn_1.050000_0.950000_1.000000_1.000000_1.000000_seed_978.pkl"
#filename = "./logs/double_pendulum_3_10_0.100000_0.001000_25_25_dyn_1.050000_0.950000_1.000000_1.000000.pkl"
filename = "./logs/double_pendulum_3_10_0.100000_0.001000_50_20_dyn_1.050000_0.950000_1.000000_1.000000.pkl"
plotter = Plotter(filename)
plotter.plot_scalar_fields(
    ["mean_return"], title="Mean return")

plotter.show()
