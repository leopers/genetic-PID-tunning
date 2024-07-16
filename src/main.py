import numpy as np
from system_simulation import SystemDynamics
from genetic_algorithm import genetic_algorithm
from visualization import animate_genetic_algorithm
from cost_functions import mse, lqr  # Import the cost functions
from pso_algorithm import pso


def main():
    # Define the system's transfer function
    num = [1]  # Numerator coefficients
    den = [1, 1.4, 1, 1]  # Denominator coefficients (s^2 + s + 1)

    system = SystemDynamics(num, den)
    time = np.linspace(0, 10, 1000)  # Define the time vector for simulation
    setpoint = np.ones_like(time)  # Define the setpoint as a constant value of 1 over time

    # Choose the cost function (mse or lqr)
    cost_function = lqr  # or mse

    # Genetic Algorithm parameters
    pop_size = 20
    num_generations = 50
    Kp_range = (0, 100)
    Ki_range = (0, 100)
    Kd_range = (0, 100)
    dt = 0.01

    # Run Genetic Algorithm
    best_pid_params, best_individuals = genetic_algorithm(
        system, setpoint, pop_size, num_generations,
        Kp_range, Ki_range, Kd_range,
        cost_function=cost_function
    )

    # Run pso algorithm
    best_pid_params_pso, best_individuals_pso = pso(system, setpoint, pop_size, num_generations, Kp_range, Ki_range, Kd_range, cost_function, dt)

    print(f"Best PID Parameters: Kp = {best_pid_params[0]}, Ki = {best_pid_params[1]}, Kd = {best_pid_params[2]}")
    print(f"Best PID PSO Parameters: Kp = {best_pid_params_pso[0]}, Ki = {best_pid_params_pso[1]}, Kd = {best_pid_params_pso[2]}")

    # Animate the Genetic Algorithm process
    animate_genetic_algorithm(best_individuals, num_generations, system)
    animate_genetic_algorithm(best_individuals_pso, num_generations, system)

if __name__ == "__main__":
    main()
