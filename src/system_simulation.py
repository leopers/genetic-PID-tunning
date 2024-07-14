import numpy as np
from scipy.signal import TransferFunction, lsim


class SystemDynamics:
    """
    A class to represent system dynamics
    """

    def __init__(self, num, den):
        self.system = TransferFunction(num, den)
        self.time = np.linspace(0, 10, 1000)

    def simulate(self, u, t):
        t_out, y_out, _ = lsim(self.system, u, t)
        return t_out, y_out

    def step_response(self, t=np.linspace(0, 10, 1000)):
        t, y = self.simulate(np.ones_like(t), t)
        return t, y
