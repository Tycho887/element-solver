import numpy as np

def shape_function(x,N):
    """
    Calculates the values of the shape functions and their derivatives at a given point x.
    x: points at which to evaluate the shape functions
    N: number of shape functions to use
    """
    PSI = np.zeros(N+1)
    dPSI = np.zeros(N+1)
    
    if N == 1:
        PSI[0] = 0.5*(1-x)
        PSI[1] = 0.5*(1+x)
        dPSI[0] = -0.5
        dPSI[1] = 0.5
    elif N == 2:
        PSI[0] = 0.5*x*(x-1)
        PSI[1] = 1-x**2
        PSI[2] = 0.5*x*(x+1)
        dPSI[0] = x-0.5
        dPSI[1] = -2*x
        dPSI[2] = x+0.5
    elif N == 3:
        PSI[0] = -0.125*x*(x-1)*(2*x-1)
        PSI[1] = 0.375*(x**2-1)*(2*x-1)
        PSI[2] = 0.375*(x**2-1)*(2*x+1)
        PSI[3] = -0.125*x*(x+1)*(2*x+1)
        dPSI[0] = -0.125*(6*x**2-6*x+1)
        dPSI[1] = 0.375*(6*x**2-3*x-2)
        dPSI[2] = 0.375*(6*x**2+3*x-2)
        dPSI[3] = -0.125*(6*x**2+6*x+1)
    else:
        raise ValueError("Only N=1,2,3 are supported.")
    
    assert np.isclose(np.sum(PSI), 1), "Shape functions do not sum to 1."
    assert np.isclose(np.sum(dPSI), 0), "Derivatives of shape functions do not sum to 0."

    return PSI, dPSI


def interpolate(f, x, N):
    """
    Interpolates the function f at the point x using N shape functions.
    f: function to interpolate
    x: point at which to interpolate
    N: number of shape functions to use
    """
    PSI, _ = shape_function(x, N)
    return np.sum(f(i) * PSI[i] for i in range(N+1))
