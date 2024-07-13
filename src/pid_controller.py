import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

class PIDController:
    def __init__(self, kp, ki, kd, sistema):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.sistema = sistema
        self.controlador = ctrl.TransferFunction([kd, kp, ki], [1, 0])
        self.malha_aberta = ctrl.series(self.controlador, self.sistema)
        self.malha_fechada = ctrl.feedback(self.malha_aberta)

    def plot(self):
        t, y = ctrl.step_response(self.malha_fechada)
        plt.figure()
        plt.plot(t, y)
        plt.xlabel('Tempo (s)')
        plt.ylabel('Amplitude')
        plt.title('Resposta ao Degrau do Sistema em Malha Fechada com Controlador PID')
        plt.grid(True)
        plt.show()

# # Exemplo de uso
# if __name__ == "__main__":
#     numerador = [1]
#     denominador = [1, 1]
#     sistema = ctrl.TransferFunction(numerador, denominador)

#     # Parâmetros do controlador PID
#     kp = 1
#     ki = 1
#     kd = 1

#     # Criar o controlador PID
#     pid_controller = PIDController(kp, ki, kd, sistema)

#     # Plotar a resposta ao degrau
#     pid_controller.plot()
