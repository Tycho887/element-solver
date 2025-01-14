import numpy as np

def gauss_quadrature(f, a, b, n):
    """
    f: function to integrate
    a: lower bound
    b: upper bound
    n: number of points to use in the Gauss-Legendre quadrature
    
    Returns: integration points x and weights w
    """
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w

def integrate(f, a, b, n):
    """
    f: function to integrate
    a: lower bound
    b: upper bound
    n: number of points to use in the Gauss-Legendre quadrature
    
    Returns: integral of f from a to b
    """
    x, w = gauss_quadrature(f, a, b, n)
    
    return np.sum(w * f((b - a) / 2 * x + (b + a) / 2)) * (b - a) / 2


if __name__ == "__main__":
    f = lambda x: np.sin(x)
    a = 0
    b = np.pi
    n = 4
    print(integrate(f, a, b, n))