import numpy as np
import control as ctrl

def criar_sistema():
    numerador = [1]
    denominador = [1, 2, 1]
    return ctrl.TransferFunction(numerador, denominador)

def simular_resposta(sistema, Kp, Ki, Kd):
    controlador_PID = ctrl.TransferFunction([Kd, Kp, Ki], [1, 0])
    sistema_malha_aberta = controlador_PID * sistema
    sistema_malha_fechada = ctrl.feedback(sistema_malha_aberta, 1)
    tempo, resposta = ctrl.step_response(sistema_malha_fechada)
    return tempo, resposta
