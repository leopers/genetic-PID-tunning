import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate_genetic_algorithm(best_individuals, num_generations):
    """
    Animate the Genetic Algorithm process
    :param best_individuals:
    :param num_generations:
    :return:
    """
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'o')
    ax.set_xlim(0, num_generations)
    ax.set_ylim(0, 10)  # Adjust this range based on your PID parameter values
    xdata, ydata = [], []

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        Kp, Ki, Kd = best_individuals[frame]
        xdata.append(frame)
        ydata.append(Kp)  # Plot Kp; you can also plot Ki and Kd
        line.set_data(xdata, ydata)
        return line,

    ani = animation.FuncAnimation(fig, update, frames=range(num_generations), init_func=init, blit=True)
    plt.xlabel('Generation')
    plt.ylabel('Kp Value')
    plt.title('Evolution of Kp Over Generations')
    plt.show()
