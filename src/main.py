import numpy as np
from system_simulation import SystemDynamics
from genetic_algorithm import genetic_algorithm
from visualization import animate_genetic_algorithm
from cost_functions import mse, lqr  # Import the cost functions
from pso_algorithm import pso
from nichols import nichols_black_tuning
from pid_controller import PIDController
import matplotlib.pyplot as plt
import copy


def main():
    # Define the system's transfer function

    # First-order system
    # num = [2]  
    # den = [5, 1] 

    # Second-order system
    # num = [16]
    # den = [1, 4, 16]

    # DC motor system
    num = [1]
    den = [0.01, 1, 0]

    system = SystemDynamics(num, den)
    time = np.linspace(0, 10, 1000)  # Define the time vector for simulation
    setpoint = np.ones_like(time)  # Define the setpoint as a constant value of 1 over time

    # Choose the cost function (mse or lqr)
    cost_function = lqr  # or mse

    # Genetic Algorithm parameters
    pop_size = 20
    num_generations = 50
    Kp_range = (0, 1000)
    Ki_range = (0, 1000)
    Kd_range = (0, 1000)
    dt = 0.01

    # Run Genetic Algorithm
    best_pid_params, best_individuals = genetic_algorithm(
        system, setpoint, pop_size, num_generations,
        Kp_range, Ki_range, Kd_range,
        cost_function=cost_function
    )

    # Run pso algorithm
    best_pid_params_pso, best_individuals_pso = pso(system, setpoint, pop_size, num_generations, Kp_range, Ki_range, Kd_range, cost_function, dt)

    # Run Zigler-Nichols method
    pid_params_zn = nichols_black_tuning(system)

    print(f"Best PID Parameters: Kp = {best_pid_params[0]}, Ki = {best_pid_params[1]}, Kd = {best_pid_params[2]}")
    print(f"Best PID PSO Parameters: Kp = {best_pid_params_pso[0]}, Ki = {best_pid_params_pso[1]}, Kd = {best_pid_params_pso[2]}")
    print(f"Zigler-Nichols PID Parameters: Kp = {pid_params_zn[0]}, Ki = {pid_params_zn[1]}, Kd = {pid_params_zn[2]}")

    # Step-response for all 3 methods best PID parameters

    # Create deep copies of the system for each PID tuning method
    system_pid = copy.deepcopy(system)
    system_pid.update_transfer_function(*best_pid_params)

    system_pid_pso = copy.deepcopy(system)
    system_pid_pso.update_transfer_function(*best_pid_params_pso)

    system_pid_zn = copy.deepcopy(system)
    system_pid_zn.update_transfer_function(*pid_params_zn)

    # Get the step responses
    t, y_pid = system_pid.step_response()
    t, y_pid_pso = system_pid_pso.step_response()
    t, y_pid_zn = system_pid_zn.step_response()

    # Plot the step response for all 3 methods best PID parameters
    y_min = min(min(y_pid), min(y_pid_pso), min(y_pid_zn))
    y_max = max(max(y_pid), max(y_pid_pso), max(y_pid_zn))

    plt.figure()
    plt.plot(t, y_pid, label='Genetic Algorithm')
    plt.plot(t, y_pid_pso, label='PSO Algorithm')
    plt.plot(t, y_pid_zn, label='Ziegler-Nichols')
    plt.xlabel('Time')
    plt.ylabel('Response')
    plt.title('Best Step Response of the System')
    plt.legend()
    plt.ylim(y_min - 0.1 * abs(y_min), y_max + 0.1 * abs(y_max))  # Add some padding to the y-axis
    plt.show()

    # Animate the Genetic Algorithm process
    animate_genetic_algorithm(best_individuals, num_generations, system)
    animate_genetic_algorithm(best_individuals_pso, num_generations, system)

if __name__ == "__main__":
    main()
