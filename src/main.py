import control as ctrl
from genetic_algorithm import GeneticAlgorithm
from pid_controller import PIDController
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

def main():
    numerador = [1]
    denominador = [1, 1, 1]
    sistema = ctrl.TransferFunction(numerador, denominador)
    
    ga = GeneticAlgorithm(sistema)
    melhor_individuo, imagens = ga.run()
    Kp, Ki, Kd = melhor_individuo
    print(f"Melhor individuo: Kp={Kp}, Ki={Ki}, Kd={Kd}")

    # Criar o controlador PID com os melhores parâmetros
    pid_controller = PIDController(Kp, Ki, Kd, sistema)

    # Criar a animação
    fig, ax = plt.subplots()
    img = None

    def update(i):
        nonlocal img
        buf = imagens[i]
        buf.seek(0)
        image = Image.open(buf)
        if img is None:
            img = ax.imshow(image)
        else:
            img.set_data(image)
        return [img]

    ani = FuncAnimation(fig, update, frames=len(imagens), blit=True)
    ani.save('evolucao_pid.gif', writer='imagemagick')

    # Plotar a resposta final
    plot_image = pid_controller.plot()
    with open("pid_response.png", "wb") as f:
        f.write(plot_image.getbuffer())

if __name__ == "__main__":
    main()
