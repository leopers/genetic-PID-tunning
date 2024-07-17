import numpy as np
import matplotlib.pyplot as plt
from system_simulation import SystemDynamics

def nichols_black_tuning(system):
    """
    Nichols-Ziegler tuning method to determine PID parameters
    :param system: TransferFunction object of the system
    :return: Tuple of PID parameters (Kp, Ki, Kd)
    """
    # Step response of the system
    t, y = system.step_response()
    
    # Calculate the ultimate gain (Ku) and the period of oscillation (Tu)
    Ku = 1
    for i in range(len(y)-1):
        if y[i] < 1 and y[i+1] > 1:
            Ku = 1 / y[i+1]
            break
    Tu = t[np.argmax(y)]
    
    # Calculate PID parameters based on Ziegler-Nichols method
    Kp = 0.6 * Ku
    Ti = 0.5 * Tu
    Td = 0.125 * Tu
    Ki = Kp / Ti
    Kd = Kp * Td

    return Kp, Ki, Kd