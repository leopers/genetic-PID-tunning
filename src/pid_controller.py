from scipy.signal import TransferFunction
import numpy as np  


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
    
    def close_loop(self, plant):
        num = np.polymul(self.system.num, plant.num)
        den = np.polymul(self.system.den, plant.den)
        open_loop_tf = TransferFunction(num, den)  # Open-loop transfer function G(s)C(s)
        closed_loop_num = open_loop_tf.num
        closed_loop_den = np.polyadd(open_loop_tf.den, open_loop_tf.num)  # 1 + G(s)C(s)
        closed_loop_tf = TransferFunction(closed_loop_num, closed_loop_den)
        return closed_loop_tf
