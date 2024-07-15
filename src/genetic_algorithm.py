import random
import numpy as np
from scipy.signal import lsim, TransferFunction
from src.pid_controller import PIDController


def initialize_population(pop_size, Kp_range, Ki_range, Kd_range):
    """
    Initialize the population with random values of Kp, Ki, and Kd
    :param pop_size:
    :param Kp_range:
    :param Ki_range:
    :param Kd_range:
    :return:
    """
    population = []
    for _ in range(pop_size):
        Kp = random.uniform(*Kp_range)
        Ki = random.uniform(*Ki_range)
        Kd = random.uniform(*Kd_range)
        population.append((Kp, Ki, Kd))
    return population


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
    open_loop_tf = system.system * pid_tf  # Open-loop transfer function G(s)C(s)
    closed_loop_num = np.polymul(open_loop_tf.num, [1])
    closed_loop_den = np.polyadd(open_loop_tf.den, open_loop_tf.num)  # 1 + G(s)C(s)
    closed_loop_tf = TransferFunction(closed_loop_num, closed_loop_den)
    t_out, y, _ = lsim(closed_loop_tf, U=np.ones_like(time), T=time)
    return y


def select_parents(population, fitnesses, num_parents):
    """
    Select the best individuals as parents based on their fitness
    :param population:
    :param fitnesses:
    :param num_parents:
    :return:
    """
    sorted_population = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    parents = [ind for ind, fit in sorted_population[:num_parents]]
    return parents


def crossover(parents, offspring_size):
    """
    Crossover the parents to create offspring
    :param parents:
    :param offspring_size:
    :return:
    """
    offspring = []
    for _ in range(offspring_size):
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        child = (
            (parent1[0] + parent2[0]) / 2,
            (parent1[1] + parent2[1]) / 2,
            (parent1[2] + parent2[2]) / 2
        )
        offspring.append(child)
    return offspring


def mutate(offspring, Kp_range, Ki_range, Kd_range, mutation_rate=0.1):
    """
    Mutate the offspring with a certain probability
    :param offspring:
    :param Kp_range:
    :param Ki_range:
    :param Kd_range:
    :param mutation_rate:
    :return:
    """
    for i in range(len(offspring)):
        if random.random() < mutation_rate:
            Kp = random.uniform(*Kp_range)
            Ki = random.uniform(*Ki_range)
            Kd = random.uniform(*Kd_range)
            offspring[i] = (Kp, Ki, Kd)
    return offspring


def create_next_generation(population, offspring):
    """
    Create the next generation by replacing the worst individuals with the offspring
    :param population:
    :param offspring:
    :return:
    """
    return population[:len(population) - len(offspring)] + offspring


def genetic_algorithm(system, setpoint, pop_size, num_generations, Kp_range, Ki_range, Kd_range, cost_function,
                      dt=0.01):
    """
    Genetic algorithm to tune the PID controller
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
    time = np.linspace(0, 10, int(10 / dt))
    population = initialize_population(pop_size, Kp_range, Ki_range, Kd_range)
    best_individuals = []

    for generation in range(num_generations):
        fitnesses = []
        for individual in population:
            Kp, Ki, Kd = individual
            pid = PIDController(Kp, Ki, Kd)
            fitness = fitness_function(system, pid, setpoint, time, dt, cost_function)
            fitnesses.append(fitness)

        parents = select_parents(population, fitnesses, num_parents=pop_size // 2)
        offspring = crossover(parents, offspring_size=pop_size // 2)
        offspring = mutate(offspring, Kp_range, Ki_range, Kd_range)
        population = create_next_generation(population, offspring)

        best_fitness = max(fitnesses)
        best_individual = population[np.argmax(fitnesses)]
        best_individuals.append(best_individual)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

    return best_individual, best_individuals
