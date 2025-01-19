import numpy as np


class Mesh:
    def __init__(self, length, elements, dimension=1):
        self.dimension = dimension
        if dimension == 1:
            self.length = length
            self.ne = elements
            self.nodes = self.ne + 1
            self.dx = length / elements
            self.x = np.linspace(0, length, self.nodes)
        elif dimension == 2:
            self.length = length
            self.ne_x, self.ne_y = elements
            self.dx = length / self.ne_x
            self.dy = length / self.ne_y
            self.x, self.y = np.meshgrid(
                np.linspace(0, length, self.ne_x + 1),
                np.linspace(0, length, self.ne_y + 1)
            )
        else:
            raise ValueError("Unsupported dimension. Only 1D and 2D are supported.")
