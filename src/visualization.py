import matplotlib.pyplot as plt

def plotar_resposta(tempo, resposta):
    plt.figure()
    plt.plot(tempo, resposta)
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.title('Resposta ao Degrau do Sistema em Malha Fechada com Controlador PID')
    plt.grid(True)
    plt.show()
