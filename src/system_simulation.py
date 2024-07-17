import numpy as np
from scipy.signal import TransferFunction, lsim
from pid_controller import PIDController


class SystemDynamics:
    """
    A class to represent system dynamics
    """

    def __init__(self, num, den):
        self.system = TransferFunction(num, den)
        self.time = np.linspace(0, 100, 10000)

    def simulate(self, u, t):
        t_out, y_out, _ = lsim(self.system, u, t)
        return t_out, y_out

    def step_response(self, t=np.linspace(0, 100, 10000)):
        t, y = self.simulate(np.ones_like(t), t)
        return t, y
    
    def update_transfer_function(self, Kp, Ki, Kd):
        pid = PIDController(Kp, Ki, Kd)
        self.system = pid.close_loop(self.system)
        