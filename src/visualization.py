import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from system_simulation import SystemDynamics
from pid_controller import PIDController
from scipy.signal import TransferFunction, lsim

def simulate_system(system, pid, time):
    """
    Simulate the system with the PID controller
    :param system:
    :param pid:
    :param time:
    :return:
    """
    pid_tf = pid.create_transfer_function()
    num = np.polymul(pid_tf.num, system.system.num)
    den = np.polymul(pid_tf.den, system.system.den)
    open_loop_tf = TransferFunction(num, den)  # Open-loop transfer function G(s)C(s)
    closed_loop_num = np.polymul(open_loop_tf.num, [1])
    closed_loop_den = np.polyadd(open_loop_tf.den, open_loop_tf.num)  # 1 + G(s)C(s)
    closed_loop_tf = TransferFunction(closed_loop_num, closed_loop_den)
    t_out, y, _ = lsim(closed_loop_tf, U=np.ones_like(time), T=time)
    return y

def animate_genetic_algorithm(best_individuals, num_generations, system):
    """
    Animate the Genetic Algorithm process
    :param best_individuals: List of tuples (Kp, Ki, Kd)
    :param num_generations: Number of generations
    :param system: Instance of SystemDynamics
    :return:
    """
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(0, 10)  # Adjust time axis limits based on your specific system
    ax.set_ylim(-2, 2)  # Adjust this range based on your system's response

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        Kp, Ki, Kd = best_individuals[frame]
        pid = PIDController(Kp, Ki, Kd)
        measurements = simulate_system(system, pid, np.linspace(0, 10, 1000))
        t = np.linspace(0, 10, 1000)  # Time vector for the step response
        line.set_data(t, measurements)
        ax.set_xlim(0, 10)   
        ax.set_ylim(-2, 2)  
        ax.set_xlabel('Time')
        ax.set_ylabel('Response')
        ax.set_title(f'Step Response at Generation {frame}')
        return line,

    ani = animation.FuncAnimation(fig, update, frames=range(len(best_individuals)), init_func=init, blit=False)
    plt.xlabel('Time')
    plt.ylabel('Response')
    plt.title('Evolution of Step Response Over Generations')
    plt.show()
