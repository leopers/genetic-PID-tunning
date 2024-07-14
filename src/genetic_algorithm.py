import random
import numpy as np
from deap import base, creator, tools, algorithms
import control as ctrl

# Função de transferência do sistema
numerador = [1]
denominador = [1, 1]
sistema = ctrl.TransferFunction(numerador, denominador)

# Função de avaliação
def avalia_pid(individual):
    Kp, Ki, Kd = individual
    controlador_PID = ctrl.TransferFunction([Kd, Kp, Ki], [1, 0])
    sistema_malha_aberta = controlador_PID * sistema
    sistema_malha_fechada = ctrl.feedback(sistema_malha_aberta, 1)
    
    tempo, resposta = ctrl.step_response(sistema_malha_fechada)
    referencia = np.ones_like(tempo)
    mse = np.mean((resposta - referencia) ** 2)
    
    return mse,

# Configurar o algoritmo genético
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", avalia_pid)

def run_ga():
    random.seed(42)
    populacao = toolbox.population(n=50)
    n_geracoes = 40

    resultado, log = algorithms.eaSimple(populacao, toolbox, cxpb=0.7, mutpb=0.2, ngen=n_geracoes, 
                                         stats=None, halloffame=None, verbose=True)
    
    return tools.selBest(resultado, k=1)[0]
