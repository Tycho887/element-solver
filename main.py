from common import integrate, shape_function, interpolate
import numpy as np

if __name__ == "__main__":
    f = lambda x: x**2 - 1
    a = 0
    b = 1
    n = 4
    print(integrate(f, a, b, n))

    N = 1
    x = 0.5
    print(shape_function(x, N))

    f = lambda x: np.sin(x)

    # Define first element

    x1 = 0
    x2 = 0.5

    # Define second element

    x3 = 0.5
    x4 = 1

    # Interpolate f at x=0.25 using N=2 shape functions

    print(interpolate(f, 0.25, N))