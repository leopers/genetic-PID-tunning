import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
import io

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
        fig, ax = plt.subplots()
        ax.plot(t, y)
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Resposta ao Degrau do Sistema em Malha Fechada com Controlador PID')
        ax.grid(True)
        
        # Salvar o plot em um buffer de memória
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        # Fechar a figura para liberar memória
        plt.close(fig)
        
        return buf

    def avalia(self):
        t, y = ctrl.step_response(self.malha_fechada)
        referencia = np.ones_like(t)
        mse = np.mean((y - referencia) ** 2)
        return mse
