from genetic_algorithm import run_ga
from pid_controller import simular_pid, plotar_resposta

def main():
    melhor_individuo = run_ga()
    Kp, Ki, Kd = melhor_individuo
    print(f"Melhor individuo: Kp={Kp}, Ki={Ki}, Kd={Kd}")

    tempo, resposta = simular_pid(Kp, Ki, Kd)
    plotar_resposta(tempo, resposta)

if __name__ == "__main__":
    main()
