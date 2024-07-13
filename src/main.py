from genetic_algorithm import run_ga
from pid_controller import simular_pid, plotar_resposta

def main():
    melhor_individuo = run_ga()
    Kp, Ki, Kd = melhor_individuo
    print(f"Melhor individuo: Kp={Kp}, Ki={Ki}, Kd={Kd}")

    numerador = [1]
    denominador = [1, 1]
    sistema = ctrl.TransferFunction(numerador, denominador)

    pid_controller = PIDController(Kp, Ki, Kd, sistema)

    pid_controller.plot()

if __name__ == "__main__":
    main()
