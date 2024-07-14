import control as ctrl
from genetic_algorithm import GeneticAlgorithm
from pid_controller import PIDController
from visualization import Visualization

def main():
    numerador = [1, 1, 1]
    denominador = [1, 1, 1, 1, 1]
    sistema = ctrl.TransferFunction(numerador, denominador)
    
    ga = GeneticAlgorithm(sistema)
    melhor_individuo, imagens = ga.run()
    Kp, Ki, Kd = melhor_individuo
    print(f"Melhor individuo: Kp={Kp}, Ki={Ki}, Kd={Kd}")

    # Criar o controlador PID com os melhores parâmetros
    pid_controller = PIDController(Kp, Ki, Kd, sistema)

    # Criar a animação
    viz = Visualization(imagens)
    viz.create_animation('evolucao_pid.gif')

    # Plotar a resposta final
    plot_image = pid_controller.plot()
    with open("pid_response.png", "wb") as f:
        f.write(plot_image.getbuffer())

if __name__ == "__main__":
    main()
