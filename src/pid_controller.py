from scipy.signal import TransferFunction


class PIDController:
    """
    A simple PID compensator. Here we work just with its parameters and transfer function, since all system methods
    are already defined in the SystemDynamics class.
    """

    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.system = TransferFunction([kd, kp, ki], [1, 0])

    def create_transfer_function(self):
        return self.system
