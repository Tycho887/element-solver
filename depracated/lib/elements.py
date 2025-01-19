from abc import ABC, abstractmethod
import numpy as np

class BasisFunctions:
    @staticmethod
    def get_basis_functions(basis_type, dimension):
        if dimension == 1:
            if basis_type == "linear":
                return {
                    "shape": lambda xi: [1 - xi, xi],
                    "shape_grad": lambda xi: [-1, 1]
                }
            elif basis_type == "quadratic":
                return {
                    "shape": lambda xi: [xi * (xi - 1) / 2, 1 - xi ** 2, xi * (xi + 1) / 2],
                    "shape_grad": lambda xi: [xi - 0.5, -2 * xi, xi + 0.5]
                }
        elif dimension == 2:
            if basis_type == "linear":
                return {
                    "shape": lambda xi, eta: [
                        (1 - xi) * (1 - eta), xi * (1 - eta), xi * eta, (1 - xi) * eta
                    ],
                    "shape_grad": lambda xi, eta: [
                        [-1 + eta, -1 + xi],
                        [1 - eta, -xi],
                        [eta, xi],
                        [-eta, 1 - xi]
                    ]
                }
        raise ValueError("Unsupported basis type or dimension")


class PDE:
    def __init__(self, pde_type, alpha):
        self.pde_type = pde_type
        self.alpha = alpha

    def get_stiffness_matrix(self, basis_grad, dx):
        if self.pde_type == "heat":
            return self.alpha / dx * np.outer(basis_grad, basis_grad)
        elif self.pde_type == "wave":
            return 1 / dx * np.outer(basis_grad, basis_grad)
        else:
            raise ValueError("Unsupported PDE type")
