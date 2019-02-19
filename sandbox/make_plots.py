


from plotter import Plotter

# Plot everything.
plotter = Plotter("./logs/double_pendulum_3_10_0.050000_0.001000_10_50.pkl")
plotter.plot_scalar_fields(["mean_return"])
plotter.show()
