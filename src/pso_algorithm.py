import random
import numpy as np
from scipy.signal import lsim, TransferFunction
from pid_controller import PIDController


class Particle:
    """
    A class to represent a particle in the PSO algorithm
    """
    def __init__(self, Kp_range, Ki_range, Kd_range):
        self.position = np.array([
            random.uniform(*Kp_range),
            random.uniform(*Ki_range),
            random.uniform(*Kd_range)
        ])
        self.velocity = np.zeros(3)
        self.best_position = np.copy(self.position)
        self.best_score = float('-inf')


def fitness_function(system, pid, setpoint, time, dt, cost_function):
    measurements = simulate_system(system, pid, time)
    return -cost_function(setpoint, measurements)  # Minimize the cost function


def simulate_system(system, pid, time):
    pid_tf = pid.create_transfer_function()
    num = np.polymul(pid_tf.num, system.system.num)
    den = np.polymul(pid_tf.den, system.system.den)
    open_loop_tf = TransferFunction(num, den)  # Open-loop transfer function G(s)C(s)
    closed_loop_num = np.polymul(open_loop_tf.num, [1])
    closed_loop_den = np.polyadd(open_loop_tf.den, open_loop_tf.num)  # 1 + G(s)C(s)
    closed_loop_tf = TransferFunction(closed_loop_num, closed_loop_den)
    t_out, y, _ = lsim(closed_loop_tf, U=np.ones_like(time), T=time)
    return y


def pso(system, time, setpoint, pop_size, num_generations, Kp_range, Ki_range, Kd_range, cost_function, dt=0.01):
    w = 0.9  # Inertia weight
    c1 = 0.6  # Cognitive (particle) weight
    c2 = 0.8  # Social (swarm) weight

    time = time
    particles = [Particle(Kp_range, Ki_range, Kd_range) for _ in range(pop_size)]
    global_best_position = None
    global_best_score = float('-inf')
    best_individuals = []

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

            # Ensure that the gains are not negative
            particle.position = np.clip(particle.position, [Kp_range[0], Ki_range[0], Kd_range[0]],
                                        [Kp_range[1], Ki_range[1], Kd_range[1]])

        best_individuals.append(global_best_position)
        print(f"Generation {generation}: Best Fitness = {global_best_score}")

    return global_best_position, best_individuals
