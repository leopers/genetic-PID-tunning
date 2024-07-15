import random
import numpy as np
from scipy.signal import lsim, TransferFunction
from pid_controller import PIDController

class Particle:
    def __init__(self, Kp_range, Ki_range, Kd_range):
        self.position = np.array([random.uniform(*Kp_range), random.uniform(*Ki_range), random.uniform(*Kd_range)])
        self.velocity = np.zeros(3)
        self.best_position = np.copy(self.position)
        self.best_score = float('-inf')

def fitness_function(system, pid, setpoint, time, dt, cost_function):
    """
    Fitness function to evaluate the performance of the PID controller
    :param system:
    :param pid:
    :param setpoint:
    :param time:
    :param dt:
    :param cost_function:
    :return:
    """
    measurements = simulate_system(system, pid, time)
    return -cost_function(setpoint, measurements)  # Minimize the cost function

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

def pso(system, setpoint, pop_size, num_generations, Kp_range, Ki_range, Kd_range, cost_function, dt=0.01):
    """
    Particle Swarm Optimization (PSO) to tune the PID controller
    :param system:
    :param setpoint:
    :param pop_size:
    :param num_generations:
    :param Kp_range:
    :param Ki_range:
    :param Kd_range:
    :param cost_function:
    :param dt:
    :return:
    """
    w = 0.5  # Inertia weight
    c1 = 2   # Cognitive (particle) weight
    c2 = 2   # Social (swarm) weight

    time = np.linspace(0, 10, int(10 / dt))
    particles = [Particle(Kp_range, Ki_range, Kd_range) for _ in range(pop_size)]
    global_best_position = None
    global_best_score = float('-inf')

    for generation in range(num_generations):
        for particle in particles:
            pid = PIDController(*particle.position)
            fitness = fitness_function(system, pid, setpoint, time, dt, cost_function)

            if fitness > particle.best_score:
                particle.best_score = fitness
                particle.best_position = np.copy(particle.position)

            if fitness > global_best_score:
                global_best_score = fitness
                global_best_position = np.copy(particle.position)

        for particle in particles:
            r1, r2 = random.random(), random.random()
            cognitive_velocity = c1 * r1 * (particle.best_position - particle.position)
            social_velocity = c2 * r2 * (global_best_position - particle.position)
            particle.velocity = w * particle.velocity + cognitive_velocity + social_velocity
            particle.position += particle.velocity

        print(f"Generation {generation}: Best Fitness = {global_best_score}")

    return global_best_position, particles
