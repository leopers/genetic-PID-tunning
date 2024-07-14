import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

class Visualization:
    def __init__(self, imagens):
        self.imagens = imagens

    def create_animation(self, output_path):
        fig, ax = plt.subplots()
        img = None

        def update(i):
            nonlocal img
            buf = self.imagens[i]
            buf.seek(0)
            image = Image.open(buf)
            if img is None:
                img = ax.imshow(image)
            else:
                img.set_data(image)
            return [img]

        ani = FuncAnimation(fig, update, frames=len(self.imagens), blit=True)
        ani.save(output_path, writer='pillow')
