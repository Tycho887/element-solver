import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Parameters
L = 1.0       # Length of the rod
T = 0.1       # Total time
alpha = 0.00001  # Thermal diffusivity
ne = 500       # Number of elements
nt = 5000      # Number of time steps
dt = T / nt   # Time step size

# Derived parameters
nodes = ne + 1                     # Total number of nodes
dx = L / ne                        # Element size
x = np.linspace(0, L, nodes)       # Node coordinates

# Basis function derivatives for 1D linear elements (constant per element)
# Reference element basis functions: N1 = 1 - xi, N2 = xi
# dN/dx = [-1/dx, 1/dx]
basis_grad = np.array([-1, 1]) / dx

# Mass and stiffness matrices for a single element
# Element Mass Matrix (M_e): Integral of N_i * N_j over element
M_e = dx / 6 * np.array([[2, 1],
                         [1, 2]])

# Element Stiffness Matrix (K_e): alpha * Integral of (dN/dx) * (dN/dx)
K_e = alpha / dx * np.outer(basis_grad, basis_grad)

# Global matrices
M = np.zeros((nodes, nodes))  # Global mass matrix
K = np.zeros((nodes, nodes))  # Global stiffness matrix

# Assembly of global matrices
for e in range(ne):
    # Indices of the current element's nodes
    indices = [e, e + 1]
    
    # Add element matrices to the global matrices
    for i in range(2):
        for j in range(2):
            M[indices[i], indices[j]] += M_e[i, j]
            K[indices[i], indices[j]] += K_e[i, j]

# Initial condition
u = np.zeros(nodes)  # Initial temperature distribution
u[int(0.4 * nodes):int(0.6 * nodes)] = 1.0  # Heat pulse in the middle

# Apply Dirichlet boundary conditions (u=0 at boundaries)
M[0, :] = M[-1, :] = 0
M[0, 0] = M[-1, -1] = 1
K[0, :] = K[-1, :] = 0
K[0, 0] = K[-1, -1] = 1

# System matrix for implicit time stepping (Backward Euler)
A = M + dt * K
A_inv = np.linalg.inv(A)  # Precompute inverse for efficiency

# Time stepping loop
for n in tqdm(range(nt), desc="Simulating Heat Equation with FEM"):
    # Right-hand side: M * u (current solution)
    b = M @ u
    
    # Solve the linear system: A * u_new = b
    u = A_inv @ b
    
    # Optional: Plotting at certain intervals
    if n % 50 == 0:
        plt.plot(x, u, label=f'Time {n*dt:.2f}')

# Final plot
plt.xlabel('Position')
plt.ylabel('Temperature')
plt.show()
