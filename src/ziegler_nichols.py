import numpy as np
from control import TransferFunction
import control
import matplotlib.pyplot as plt

def find_kpu_response(system, kpu):
    """
    Find the response of the system with a given Kp value.
    :param system: SystemDynamics object of the system.
    :param kpu: Proportional gain value.
    :return: Time array and output response array.
    """
    tf_sys = system.system
    tf_pid = TransferFunction(np.array([kpu]), [1])
    tf_mul = tf_sys * tf_pid
    tf_sys_pid = tf_mul.feedback()

    t, y = control.step_response(tf_sys_pid)
    return t, y

def calculate_oscillation_period(t, y):
    """
    Calculate the period of oscillations (Tu).
    :param t: Time array.
    :param y: Output response array.
    :return: Period of oscillations (Tu).
    """
    mean_y = 1
    zero_crossings = np.where(np.diff(np.sign(y - mean_y)))[0]

    if len(zero_crossings) < 2:
        raise ValueError("Not enough zero crossings to calculate the oscillation period.")
    Tu = 2*np.mean(np.diff(t[zero_crossings]))
    return Tu

def ziegler_nichols_tuning(system, kpu):
    """
    Ziegler-Nichols tuning method to determine PID parameters.
    :param system: SystemDynamics object of the system.
    :param initial_Kp: Initial value of Kp to start the search.
    :param step_Kp: Increment step for Kp.
    :param max_Kp: Maximum value of Kp to search.
    :param threshold: Threshold for detecting divergence.
    :return: Tuple of PID parameters (Kp, Ki, Kd).
    """
    t, y = find_kpu_response(system, kpu)
    Tu = calculate_oscillation_period(t, y)
    
    # Calculate PID parameters based on Ziegler-Nichols method
    Kp = 0.6 * kpu
    Ti = 0.5 * Tu
    Td = 0.125 * Tu
    Ki = Kp / Ti
    Kd = Kp * Td

    return Kp, Ki, Kd
