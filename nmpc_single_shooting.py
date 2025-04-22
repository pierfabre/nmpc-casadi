import casadi as ca
import numpy as np
from cartpole import Cartpole
import time
import matplotlib.pyplot as plt

# ========== Parameters ==========
dt = # TODO
N = # TODO

# Simulation setup
x0 = np.array([0.0, 0.0, np.pi, 0.0])      # cart at 0, pole inverted
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

# ========== Integrator ==========
x = ca.SX.sym('x', cartpole.nx)
u = ca.SX.sym('u', cartpole.nu)

xdot = cartpole.dynamics(x, u)

dae = {'x': x, 'p': u, 'ode': xdot}
integrator_opts = {
    'tf': dt,           # Integration dt
    'simplify': True,
    'number_of_finite_elements': 4  # 4 steps = RK4
}
integrator = ca.integrator('integrator', 'rk', dae, integrator_opts)

# ========== Optimization Problem ==========
"""
Hint: Use nmpc_multiple_shooting.py
"""
opti = ca.Opti()

# TODO

# ========== Solver ==========
opti.solver("ipopt")

# ========== Simulation ==========
x_hist = [x0]
u_hist = []
solve_times = []

current_x = x0

U_sol = np.zeros((nu, N))

for t in range(sim_steps):
    # Set parameters value
    # TODO

    # Set variables initial guess (warm-start)
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
    # TODO

    # Send the action to the cartpole
    u = # TODO
    current_x = cartpole.sim(current_x, u)
    
    u_hist.append(u)
    x_hist.append(current_x)
    
    # Prepare warm-start (shift and extend)
    """Hint: use np.hstack"""
    # TODO

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