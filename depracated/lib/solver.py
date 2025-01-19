from tqdm import tqdm
import numpy as np

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
        elif self.mesh.dimension == 2:
            self.assemble_2d()
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
        # Extend for other boundary conditions

    def solve(self, u0):
        self.u = u0
        A = self.M + self.dt * self.K
        A_inv = np.linalg.inv(A)

        for n in tqdm(range(self.nt), desc="Time-stepping"):
            b = self.M @ self.u
            self.u = A_inv @ b
