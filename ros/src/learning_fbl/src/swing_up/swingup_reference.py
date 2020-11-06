#!/usr/bin/env python

import numpy as np
from fbl_core.reference_generator import ReferenceGenerator

class SwingupReference(ReferenceGenerator):
    def __init__(self, nominal_dynamics=None, max_path_length = 1000):
        self._nominal_dynamics = nominal_dynamics
        self.max_path_length = max_path_length
        self._time_step = self._nominal_dynamics.time_step

    def __call__(self, x):
        """ Return a reference trajectory which starts from x """
        return self._generate_reference_traj(x, self.max_path_length)


    def _generate_reference_traj(self, x0, max_path_length):
        """ I ripped this directly from Tyler's code in sandbox and only modified variable names, etc"""


        MAX_CONTINUOUS_TIME_FREQ = 0.1
        MAX_DISCRETE_TIME_FREQ = MAX_CONTINUOUS_TIME_FREQ * self._time_step

        A, B, C = self._nominal_dynamics.linearized_system()

        linsys_xdim=A.shape[0]
        linsys_udim=B.shape[1]

        # Initial y.
        y0 = self._nominal_dynamics.linearized_system_state(x0)

        y = np.empty((linsys_xdim, max_path_length))
        for ii in range(linsys_xdim):
            y[ii, :] = np.linspace(
                0, max_path_length * self._time_step,
                max_path_length)
            y[ii, :] = y0[ii, 0] + 1.0 * np.random.uniform() * (1.0 - np.cos(
                2.0 * np.pi * MAX_DISCRETE_TIME_FREQ * \
                np.random.uniform() * y[ii, :])) #+ 0.1 * np.random.normal()

        # Ensure that y ref starts at y0.
        assert(np.allclose(y[:, 0].flatten(), y0.flatten(), 1e-5))

        return np.split(y, indices_or_sections=max_path_length, axis=1)