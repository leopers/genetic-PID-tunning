import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from system_simulation import SystemDynamics
from pid_controller import PIDController
from scipy.signal import TransferFunction, lsim

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
        closed_loop_tf = pid.close_loop(system.system)
        t, y, _ = lsim(closed_loop_tf, U=np.ones_like(system.time), T=system.time)
        line.set_data(t, y)
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
