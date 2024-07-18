from typing import Any
import os
from utils import generate_gen
from calc_fitness import evaluate_fitness, evaluate_mutation_fitness
from system_simulation import SystemDynamics

import control
from control import TransferFunction
from tqdm import tqdm
from random import random
import numpy as np 
import copy
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    """
    Class implementing a Genetic Algorithm for PID controller optimization.
    
    Attributes:
    system (SystemDynamics): The system dynamics.
    n_var (int): Number of variables (genes).
    n_bit (int): Number of bits per variable.
    ra (float): The lower bound of the range.
    rb (float): The upper bound of the range.
    population_size (int): Size of the population.
    target (float): Minimum fitness target for termination.
    """
    def __init__(self, system, n_var, n_bit, ra, rb, population_size, minimum_target = 75) -> None:
        """
        Initialize the GeneticAlgorithm with the given parameters.
        
        Parameters:
        system (SystemDynamics): The system to be controlled.
        n_var (int): Number of variables (genes).
        n_bit (int): Number of bits per variable.
        ra (float): The lower bound of the range.
        rb (float): The upper bound of the range.
        population_size (int): Size of the population.
        minimum_target (float): Minimum fitness target for termination. Default is 75.
        """
        self.num = system.system.num
        self.den = system.system.den
        self.n_var = n_var
        self.n_bit = n_bit
        self.ra = ra 
        self.rb = rb
        self.population_size = population_size
        self.target = minimum_target

        # Lists to store the generation and the parameters
        self.kp_list = []
        self.ki_list = []
        self.kd_list = []
        self.fitness_list = []

    def create_population(self):
        """
        Create the initial population.

        Returns:
        list: A list of dictionaries, each representing an individual in the population with its genes, fitness, and chromosome.
        """
        print("Create the population")
        population = []
        for _ in tqdm(range(self.population_size)):
            gen, chromosome = generate_gen(self.n_var, self.n_bit, self.ra, self.rb)
            fitness = evaluate_fitness(self.num, self.den, gen)
            population.append(
                {
                    "gen": gen,
                    "fitness": fitness,
                    "chromosome": chromosome
                }
            )
        return population
    
    def selection(self, population):
        """
        Select two parents based on their fitness.

        Parameters:
        population (list): The current population.

        Returns:
        tuple: A tuple containing the two selected parents.
        """
        population_cp = copy.deepcopy(population)
        fitness = []
        for pop in population_cp:
            fitness.append(pop["fitness"])
        index = np.argmax(fitness)
        parent1 = population_cp[index]

        population_cp[index] = {}
        fitness[index] = 0

        index = np.argmax(fitness)
        parent2 = population_cp[index]

        return parent1, parent2

    def crossover(self, parent1, parent2):
        """
        Perform crossover between two parents to produce two children.

        Parameters:
        parent1 (dict): The first parent.
        parent2 (dict): The second parent.

        Returns:
        tuple: A tuple containing the two children.
        """
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        slice_index = len(parent1["chromosome"]) // 2

        child1["chromosome"][:slice_index] = parent2["chromosome"][:slice_index]
        child2["chromosome"][:slice_index] = parent1["chromosome"][:slice_index]

        return child1, child2
    
    def mutation(self, child, mutation_rate):
        """
        Perform mutation on a child.

        Parameters:
        child (dict): The child to be mutated.
        mutation_rate (float): The mutation rate.

        Returns:
        dict: The mutated child.
        """
        mutant = child
        mutator = {
            0: 1, 
            1: 0
        }
        chromosome = child["chromosome"]
        for i in range(len(chromosome)):
            if random() < mutation_rate:
                chromosome[i] = mutator[chromosome[i]]

        mutant["chromosome"] = chromosome

        return mutant
    
    def regeneration(self, children, population):
        """
        Replace the least fit individuals in the population with the children.

        Parameters:
        children (list): The children to be added to the population.
        population (list): The current population.

        Returns:
        list: The new population.
        """
        fitness = []
        for pop in population:
            fitness.append(pop["fitness"])

        for i in range(len(children)):
            index = np.argmin(fitness)
            population[index] = children[i]
            fitness[index] = np.inf

        return population
    
    def termination(self, population):
        """
        Check if the termination condition is met.

        Parameters:
        population (list): The current population.

        Returns:
        tuple: A tuple containing the best individual and a boolean indicating whether to continue or terminate.
        """
        best, _ = self.selection(population)

        if best["fitness"] > self.target:
            loop = False
        else: 
            loop = True

        return best, loop
    
    def display_out(self, pop, generation):
        """
        Display the optimization results for the current generation.

        Parameters:
        pop (dict): The best individual in the current generation.
        generation (int): The current generation number.
        """
        print("##### Optimizing PID using Genetic Algorithm #####")
        print(f"* Generation: {generation}")
        print(f"* KP        : {pop['gen'][0]}")
        print(f"* KI        : {pop['gen'][1]}")
        print(f"* KD        : {pop['gen'][2]}")
        print(f"* Fitness   : {pop['fitness']}")

    def get_PID(self, num, den, pop):
        """
        Get the PID parameters from the best individual and display the step response.

        Parameters:
        num (list): Numerator coefficients of the transfer function.
        den (list): Denominator coefficients of the transfer function.
        pop (dict): The best individual containing the PID parameters.
        """
        tf_sys = TransferFunction(num, den)
        best_gen_pid = pop["gen"]
        kp, ki, kd = best_gen_pid[0], best_gen_pid[1], best_gen_pid[2]

        return kp, ki, kd
        
    def __call__(self, mutation_rate=0.5, *args: Any, **kwds: Any) -> Any:
        """
        Run the Genetic Algorithm to optimize PID parameters.

        Parameters:
        mutation_rate (float): The mutation rate. Default is 0.5.

        Returns:
        Any: The result of the optimization.
        """
        population = self.create_population()

        looping = True
        generation = 0

        while looping:

            parent1, parent2 = self.selection(population)

            child1, child2 = self.crossover(parent1, parent2)

            mutation1 = self.mutation(child1, mutation_rate)
            mutation2 = self.mutation(child2, mutation_rate)

            fitness_mutation1, gen1 = evaluate_mutation_fitness(mutation1, self.num, self.den, self.n_var, self.n_bit, self.ra, self.rb)
            fitness_mutation2, gen2 = evaluate_mutation_fitness(mutation2, self.num, self.den, self.n_var, self.n_bit, self.ra, self.rb)

            mutation1["gen"] = gen1
            mutation2["gen"] = gen2 
            mutation1["fitness"] = fitness_mutation1
            mutation2["fitness"] = fitness_mutation2

            population = self.regeneration([mutation1, mutation2], population)
            best, looping = self.termination(population)
            self.display_out(best, generation)

            # Store the generation and the parameters
            self.kp_list.append(best["gen"][0])
            self.ki_list.append(best["gen"][1])
            self.kd_list.append(best["gen"][2])
            self.fitness_list.append(best["fitness"])

            generation += 1

        return self.get_PID(self.num, self.den, best)

    def plot_evolution(self):
        """
        Plot the evolution of the PID coefficients and fitness over generations
        and save each plot as a separate file in the results directory.
        """
        results_dir = '../results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        generations = range(len(self.kp_list))

        # Plot Kp evolution
        plt.figure()
        plt.plot(generations, self.kp_list, label='Kp', color='r')
        plt.xlabel('Generation')
        plt.ylabel('Kp')
        plt.title('Evolution of Kp')
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, 'Kp_evolution.png'))
        plt.close()

        # Plot Ki evolution
        plt.figure()
        plt.plot(generations, self.ki_list, label='Ki', color='g')
        plt.xlabel('Generation')
        plt.ylabel('Ki')
        plt.title('Evolution of Ki')
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, 'Ki_evolution.png'))
        plt.close()

        # Plot Kd evolution
        plt.figure()
        plt.plot(generations, self.kd_list, label='Kd', color='b')
        plt.xlabel('Generation')
        plt.ylabel('Kd')
        plt.title('Evolution of Kd')
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, 'Kd_evolution.png'))
        plt.close()

        # Plot fitness evolution
        plt.figure()
        plt.plot(generations, self.fitness_list, label='Fitness', color='m')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Evolution of Fitness')
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, 'Fitness_evolution.png'))
        plt.close()