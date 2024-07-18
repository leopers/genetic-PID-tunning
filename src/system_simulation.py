import numpy as np
from control import TransferFunction
from pid_controller import PIDController


class SystemDynamics:
    """
    A class to represent system dynamics
    """

    def __init__(self, num, den):
        self.system = TransferFunction(num, den)
        