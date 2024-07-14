import random
import numpy as np
from deap import base, creator, tools, algorithms
import control as ctrl
from pid_controller import PIDController

class GeneticAlgorithm:
    def __init__(self, sistema, n_pop=50, n_gen=40):
        self.sistema = sistema
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.toolbox = self.setup_toolbox()

    def setup_toolbox(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, 0, 10)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", self.avalia_pid)
        
        return toolbox

    def avalia_pid(self, individual):
        Kp, Ki, Kd = individual
        controlador_PID = PIDController(Kp, Ki, Kd, self.sistema)
        mse = controlador_PID.avalia()
        return mse,

    def run(self):
        random.seed(42)
        populacao = self.toolbox.population(n=self.n_pop)
        
        imagens = []

        def save_images(gen):
            melhor_individuo = tools.selBest(populacao, k=1)[0]
            Kp, Ki, Kd = melhor_individuo
            controlador_PID = PIDController(Kp, Ki, Kd, self.sistema)
            imagem = controlador_PID.plot()
            imagens.append(imagem)

        for gen in range(self.n_gen):
            algorithms.varAnd(populacao, self.toolbox, cxpb=0.7, mutpb=0.2)
            fits = self.toolbox.map(self.toolbox.evaluate, populacao)
            for fit, ind in zip(fits, populacao):
                ind.fitness.values = fit
            populacao = self.toolbox.select(populacao, len(populacao))
            save_images(gen)

        melhor_individuo = tools.selBest(populacao, k=1)[0]
        return melhor_individuo, imagens
