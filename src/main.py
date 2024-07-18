from system_simulation import SystemDynamics
from genetic_algorithm import GeneticAlgorithm
from pid_controller import PIDController
from plotting import plot_PID
from ziegler_nichols import ziegler_nichols_tuning
import matplotlib.pyplot as plt

def compare_tuning():
    # Define the system transfer function
    num = [20]
    den = [1, 32, 140, 0]
    system = SystemDynamics(num, den)
    
    # Genetic Algorithm Tuning
    n_var = 3                       # Kp, Ki, Kd
    n_bit = 10
    ra = 100                        # Upper bound
    rb = 0                          # Lower bound
    population = 100
    minimum_target = 82

    ga_tuner = GeneticAlgorithm(system, n_var, n_bit, ra, rb, population, minimum_target)
    kp_ga, ki_ga, kd_ga = ga_tuner()
    pid_ga = PIDController(kp_ga, ki_ga, kd_ga)

    # Ziegler-Nichols Tuning
    kpu = 224  # Ultimate gain, obtained via Root Locus Calculation (Steady state oscilations)
    kp_zn, ki_zn, kd_zn = ziegler_nichols_tuning(system, kpu)
    pid_zn = PIDController(kp_zn, ki_zn, kd_zn)

    # Get the step responses
    t_ga, y_ga = plot_PID(system, pid_ga)
    t_zn, y_zn = plot_PID(system, pid_zn)

    # Plot results
    plt.figure()

    plt.plot(t_ga, y_ga, label='Genetic Algorithm Tuning')
    plt.plot(t_zn, y_zn, label='Ziegler-Nichols Tuning')
    
    plt.title('Comparison of PID Tuning Methods')
    plt.xlabel('Time')
    plt.ylabel('Response')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    compare_tuning()
