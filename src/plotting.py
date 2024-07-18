from venv import create
from matplotlib import pyplot as plt
from control import TransferFunction
import numpy as np
import control

def plot_PID(sis, pid):
        """
        Get the PID parameters from the best individual and display the step response.

        Parameters:
        sis (SystemDynamics): The system dynamics object.
        pid (PIDController): The PID controller object.
        """
        tf_sys = sis.system
        kp, ki, kd = pid.kp, pid.ki, pid.kd

        tf_pid = pid.create_transfer_function()

        tf_mul = tf_sys * tf_pid
        tf_sys_pid = tf_mul.feedback()
     
        result = control.step_info(tf_sys_pid)
        
        print("### Best of PID Parameters ###")
        print(f"---> KP: {kp} - KI: {ki} - KD: {kd} <---")
        print("@@ Result:")
        print(f"* Rise Time         : {result['RiseTime']}")
        print(f"* Overshoot         : {result['Overshoot']}")
        print(f"* SettlingTime      : {result['SettlingTime']}")
        print(f"* SteadyState       : {result['SteadyStateValue']}")

        t, y = control.step_response(tf_sys_pid)
        return t, y

