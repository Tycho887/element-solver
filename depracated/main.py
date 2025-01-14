import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm

# Parameters
L = 1.0       # Length of the rod
T = 0.1       # Total time
alpha = 0.01  # Thermal diffusivity
nx = 50       # Number of spatial points
nt = 500      # Number of time steps
dx = L / (nx - 1)  # Spatial step size
dt = T / nt       # Time step size

# Stability criterion
if alpha * dt / dx**2 > 0.5:
    raise ValueError("Stability condition not met, reduce dt or increase dx.")

# Discretized domain
x = np.linspace(0, L, nx)
u = np.zeros(nx)  # Initial temperature distribution
u_new = np.zeros(nx)

# Initial condition
u[int(0.4 * nx):int(0.6 * nx)] = 1.0  # Heat pulse in the middle

# Time stepping loop with tqdm
for n in tqdm(range(nt), desc="Simulating Heat Equation"):
    for i in range(1, nx - 1):
        u_new[i] = u[i] + alpha * dt / dx**2 * (u[i+1] - 2*u[i] + u[i-1])
    u[:] = u_new  # Update solution

    # Optional: Plotting at certain intervals
    if n % 50 == 0:
        plt.plot(x, u, label=f'Time {n*dt:.2f}')

# Final plot
plt.xlabel('Position')
plt.ylabel('Temperature')
plt.legend()
plt.show()
