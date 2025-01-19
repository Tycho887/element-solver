import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import toml

# Mesh class
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

# Basis functions
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

# PDE class
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

# Solver class
class Solver:
    def __init__(self, mesh, basis, pde, time_config, bc_config):
        self.mesh = mesh
        self.basis = basis
        self.pde = pde
        self.dt = time_config["total_time"] / time_config["time_steps"]
        self.nt = time_config["time_steps"]
        self.bc_type = bc_config["type"]
        self.bc_values = bc_config["value"]

        # Initialize global matrices
        self.M = None
        self.K = None
        self.u = None

    def assemble_matrices(self):
        if self.mesh.dimension == 1:
            self.assemble_1d()
        else:
            raise ValueError("Unsupported mesh dimension")

    def assemble_1d(self):
        M_e = self.mesh.dx / 6 * np.array([[2, 1], [1, 2]])
        basis_grad = np.array(self.basis["shape_grad"](None)) / self.mesh.dx
        K_e = self.pde.get_stiffness_matrix(basis_grad, self.mesh.dx)

        self.M = np.zeros((self.mesh.nodes, self.mesh.nodes))
        self.K = np.zeros((self.mesh.nodes, self.mesh.nodes))

        for e in range(self.mesh.ne):
            indices = [e, e + 1]
            for i in range(2):
                for j in range(2):
                    self.M[indices[i], indices[j]] += M_e[i, j]
                    self.K[indices[i], indices[j]] += K_e[i, j]

    def apply_boundary_conditions(self):
        if self.bc_type == "dirichlet":
            self.M[0, :] = self.M[-1, :] = 0
            self.M[0, 0] = self.M[-1, -1] = 1
            self.K[0, :] = self.K[-1, :] = 0
            self.K[0, 0] = self.K[-1, -1] = 1

    def solve(self, u0):
        self.u = u0
        A = self.M + self.dt * self.K
        A_inv = np.linalg.inv(A)

        solution_over_time = [self.u.copy()]

        for _ in tqdm(range(self.nt), desc="Time-stepping"):
            b = self.M @ self.u
            self.u = A_inv @ b
            solution_over_time.append(self.u.copy())

        return solution_over_time

# Initialization function
def initialize_solution(mesh, u, init_config):
    if init_config["type"] == "pulse":
        start = int(init_config["start_fraction"] * len(mesh.x))
        end = int(init_config["end_fraction"] * len(mesh.x))
        u[start:end] = init_config["value"]
    elif init_config["type"] == "sinusoidal":
        u = np.sin(np.pi * mesh.x / mesh.length)
    else:
        raise ValueError(f"Unsupported initial condition type: {init_config['type']}")
    return u

# Main function
def main():
    config = toml.load("config.toml")
    mesh = Mesh(config["domain"]["length"], config["domain"]["elements"], dimension=1)
    basis = BasisFunctions.get_basis_functions(config["basis"]["type"], 1)
    pde = PDE(config["pde"]["type"], config["pde"].get("diffusivity", 0.00001))

    solver = Solver(mesh, basis, pde, config["time"], config["boundary_conditions"])
    solver.assemble_matrices()
    solver.apply_boundary_conditions()

    u0 = np.zeros(mesh.nodes)
    u0 = initialize_solution(mesh, u0, config["initial_condition"])

    solutions = solver.solve(u0)

    # Create animation
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(0, mesh.length)

    ax.set_ylim(0, np.max(np.max(solutions)))

    def update(frame):
        line.set_data(mesh.x, solutions[frame])
        return line,

    ani = FuncAnimation(fig, update, frames=len(solutions), blit=True)
    ani.save("output.mp4", fps=30)
    plt.show()

if __name__ == "__main__":
    main()
