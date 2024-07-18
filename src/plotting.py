import os
from matplotlib import pyplot as plt
from control import TransferFunction
import numpy as np
import control

def plot_PID(sis, pid, filename='pid_results.txt'):
    """
    Get the PID parameters from the best individual and display the step response.

    Parameters:
    sis (SystemDynamics): The system dynamics object.
    pid (PIDController): The PID controller object.
    filename (str): The name of the file to save the results.
    """
    # Ensure the results directory exists
    results_dir = '../results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Full file path
    filepath = os.path.join(results_dir, filename)
    
    tf_sys = sis.system
    kp, ki, kd = pid.kp, pid.ki, pid.kd

    tf_pid = pid.create_transfer_function()

    tf_mul = tf_sys * tf_pid
    tf_sys_pid = tf_mul.feedback()
     
    result = control.step_info(tf_sys_pid)
    
    output = [
        "### Best of PID Parameters ###",
        f"---> KP: {kp} - KI: {ki} - KD: {kd} <---",
        "@@ Result:",
        f"* Rise Time         : {result['RiseTime']}",
        f"* Overshoot         : {result['Overshoot']}",
        f"* SettlingTime      : {result['SettlingTime']}",
        f"* SteadyState       : {result['SteadyStateValue']}"
    ]
    
    with open(filepath, 'w') as f:
        for line in output:
            f.write(line + '\n')
    
    for line in output:
        print(line)

    t, y = control.step_response(tf_sys_pid)
    return t, y