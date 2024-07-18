import numpy as np
from control import TransferFunction
import control

def evaluate_fitness(num, den, gen):
    """
    Calculate the fitness of a PID controller.

    Parameters:
        num (list): Numerator coefficients of the system transfer function.
        den (list): Denominator coefficients of the system transfer function.
        gen (list): PID parameters.

    Returns:
        float: Fitness value.
    """
    tf_sys = TransferFunction(num, den)
    kp, ki, kd = gen[0], gen[1], gen[2]
    tf_pid = TransferFunction([kd, kp, ki], [1, 0])
    tf_mul = tf_sys * tf_pid
    tf_sys_pid = tf_mul.feedback()

    try:
        result = control.step_info(tf_sys_pid)
        ess = result["SteadyStateValue"]
        rise_time = result["RiseTime"]
        settling_time = result["SettlingTime"]
        overshoot = result["Overshoot"]

        fitness_1 = 1/(rise_time + 1) * 100
        fitness_2 = 1/(ess + 0.1) * 100
        fitness_3 = 1/(overshoot + 1) * 100 if ess != 0 else 100
        fitness_4 = 1/(settling_time + 0.01) * 100 if settling_time > 10 else 100

        fitness = (fitness_1 + fitness_2 + fitness_3 + fitness_4) / 4
        return fitness

    except BaseException as e:
        print(f"Kp: {kp} - Ki: {ki} - Kd: {kd}")
        print(tf_sys_pid)
        print(e)
        return 0        
    
def evaluate_mutation_fitness(mutant, num, den, n_var, n_bit, lb, ub):
    """
    Calculate the fitness of a mutated PID controller.

    Parameters:
        mutant (dict): Mutant individual.
        num (list): Numerator coefficients of the system transfer function.
        den (list): Denominator coefficients of the system transfer function.
        n_var (int): Number of variables.
        n_bit (int): Number of bits per variable.
        lb (float): Lower bound for parameter values.
        ub (float): Upper bound for parameter values.

    Returns:
        tuple: Fitness value and generated genes.
    """
    gen = []
    chromosome = mutant["chromosome"]
    for i in range(n_var):
        kr = chromosome[n_bit * i: n_bit * (i + 1)]
        x = np.dot(kr, [2**(n_bit - j - 1) for j in range(n_bit)])
        x = x / ((2**n_bit) - 1)
        gen.append(lb + (ub - lb) * x)

    fitness = evaluate_fitness(num, den, gen)
    return fitness, gen