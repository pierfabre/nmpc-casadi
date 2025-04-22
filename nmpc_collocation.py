import casadi as ca
import numpy as np
from cartpole import Cartpole
import time
import matplotlib.pyplot as plt

# ========== Parameters ==========
dt = # TODO
N = # TODO

# Simulation setup
x0 = np.array([0.0, 0.0, 3.0, 0.0])      # cart at 0, pole inverted
x_target = np.array([0.0, 0.0, 0.0, 0.0])  # cart at 0, pole upright    
sim_time = 5                   # Total simulation time in seconds
sim_steps = int(sim_time / dt)     # Number of control intervals

# Cost weights
Q = # TODO
R = # TODO

# Cartpole
cartpole = Cartpole(dt)

nx = cartpole.nx
nu = cartpole.nu

# ========== Collocation ==========
d = 3  # Degree of interpolating polynomial
tau_root = [0] + ca.collocation_points(d, 'radau')
C = np.zeros((d+1, d+1))
D = np.zeros(d+1)
B = np.zeros(d+1)

# Coefficients for collocation equations
for j in range(d+1):
    # Construct Lagrange polynomials
    L = np.poly1d([1])
    for r in range(d+1):
        if r != j:
            L *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])
    D[j] = L(1.0)
    lp = np.polyder(L)  
    for r in range(d+1):
        C[j, r] = lp(tau_root[r])
    integral = np.polyint(L)
    B[j] = integral(1.0)

# ========== Optimization Problem ==========
"""
Hint: Use nmpc_multiple_shooting.py
"""
opti = ca.Opti()

# Decision variables
X = # TODO
U = # TODO
Xc = [[opti.variable(nx) for _ in range(d)] for _ in range(N)]

# Parameters
X0 = # TODO
X_ref = # TODO

# Initial constraint
# TODO

cost = 0

for k in range(N):
    x_k = X[:, k]
    u_k = U[:, k]

    # Input constraint
    # TODO

    # State constraint
    # TODO

    # Collocation equations
    Xk_end = D[0] * x_k
    for j in range(d):
        xp = C[0, j+1] * x_k
        for r in range(d):
            xp += C[r+1, j+1] * Xc[k][r]
        fk = cartpole.dynamics(Xc[k][j], u_k)
        # Collocation constraints
        opti.subject_to(dt * fk == xp)
        Xk_end += D[j+1] * Xc[k][j]

        # Running cost
        cost += ca.mtimes([(Xc[k][j] - X_ref).T, Q, (Xc[k][j] - X_ref)]) + ca.mtimes([u_k.T, R, u_k])

    # Dynamics constraint
    # TODO

# Terminal cost
# TODO

opti.minimize(cost)

# ========== Solver ==========
opts = {
    "ipopt": {
        "warm_start_init_point": "yes",
    }
}

opti.solver("ipopt", opts)

# ========== Simulation ==========
x_hist = [x0]
u_hist = []
solve_times = []

current_x = x0

# Initialize the initial guess (optional)
X_warm_start = # TODO
U_warm_start = # TODO

# Set initial guess
# TODO

for t in range(sim_steps):
    # Set parameters value
    # TODO
    
    start_time = time.time()
    # Solve NMPC
    try:
        sol = opti.solve()
    except RuntimeError as e:
        print(f"Solver failed at time {t * dt:.2f}s: {str(e)}")
        break
    solve_times.append(time.time() - start_time)

    # Get solution from solver
    X_sol = # TODO
    U_sol = # TODO
    
    # Send the action to the cartpole
    u = # TODO
    current_x = cartpole.sim(current_x, u)
    
    u_hist.append(u)
    x_hist.append(current_x)

x_hist = np.array(x_hist)
u_hist = np.vstack(u_hist)

# ========== Plot ==========
cartpole.plot(x_hist, u_hist)
cartpole.animate(x_hist)

# ========== Computation time ==========
print(f"Average NMPC solve time: {np.mean(solve_times)*1000:.2f} ms")

times = np.arange(len(solve_times)) * cartpole.dt
plt.figure()
plt.scatter(times, solve_times)
plt.yscale('log')
plt.xlabel('Timestep')
plt.ylabel('NMPC Solve Time (s)')
plt.title('NMPC Computation Time per Step')
plt.grid(True, which='both', linestyle='--', linewidth=0.5) 
plt.show()