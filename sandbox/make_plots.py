


from plotter import Plotter

# Plot everything.
plotter1 = Plotter("./logs/double_pendulum_3_10_0.050000_0.001000_10_50.pkl")
plotter1.plot_scalar_fields(
    ["mean_return"], title="double_pendulum_3_10_0.050000_0.001000_10_50")
plotter1.show()

plotter2 = Plotter("./logs/double_pendulum_3_10_0.020000_0.001000_25_25.pkl")
plotter2.plot_scalar_fields(
    ["mean_return"], title="double_pendulum_3_10_0.020000_0.001000_25_25")
plotter2.show()

plotter3 = Plotter("./logs/double_pendulum_3_10_0.050000_0.001000_25_25.pkl")
plotter3.plot_scalar_fields(
    ["mean_return"], title="double_pendulum_3_10_0.050000_0.001000_25_25")
plotter3.show()

plotter4 = Plotter("./logs/double_pendulum_3_10_0.050000_0.010000_10_50.pkl")
plotter4.plot_scalar_fields(
    ["mean_return"], title="double_pendulum_3_10_0.050000_0.010000_10_50")
plotter4.show()

plotter5 = Plotter("./logs/double_pendulum_3_10_0.100000_0.001000_25_25.pkl")
plotter5.plot_scalar_fields(
    ["mean_return"], title="double_pendulum_3_10_0.100000_0.001000_25_25")
plotter5.show()

plotter6 = Plotter("./logs/double_pendulum_5_10_0.050000_0.001000_25_25.pkl")
plotter6.plot_scalar_fields(
    ["mean_return"], title="double_pendulum_5_10_0.050000_0.001000_25_25")
plotter6.show()
