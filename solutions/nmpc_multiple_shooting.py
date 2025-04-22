import casadi as ca
import numpy as np
from cartpole import Cartpole
import time
import matplotlib.pyplot as plt

# ========== Parameters ==========
dt = 0.1
N = 20

# Simulation setup
x0 = np.array([0.0, 0.0, np.pi, 0.0])      # cart at 0, pole inverted
x_target = np.array([0.0, 0.0, 0.0, 0.0])  # cart at 0, pole upright    
sim_time = 5                   # Total simulation time in seconds
sim_steps = int(sim_time / dt)     # Number of control intervals

# Cost weights
Q = np.diag([10, 1, 50, 1])
R = np.diag([0.1])

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
opti = ca.Opti()

# Decision variables
X = opti.variable(nx, N+1)
U = opti.variable(nu, N)

# Parameters
X0 = opti.parameter(nx)
X_ref = opti.parameter(nx)

# Initial constraint
opti.subject_to(X[:, 0] == X0)

cost = 0

for k in range(N):
    x_k = X[:, k]
    u_k = U[:, k]

    # Input constraint
    opti.subject_to(opti.bounded(-10, u_k, 10))

    # State constraint
    opti.subject_to(opti.bounded(-1.0, x_k[0], 1.0))

    # Running cost
    cost += ca.mtimes([(x_k - X_ref).T, Q, (x_k - X_ref)]) + ca.mtimes([u_k.T, R, u_k])

    # Integrate
    x_next = integrator(x0=x_k, p=u_k)['xf']
    
    # Dynamics constraint
    opti.subject_to(X[:, k+1] == x_next)

# Terminal cost
cost += ca.mtimes([(X[:, N] - X_ref).T, Q, (X[:, N] - X_ref)])

opti.minimize(cost)

# ========== Solver ==========
opti.solver("ipopt")

# ========== Simulation ==========
x_hist = [x0]
u_hist = []
solve_times = []

current_x = x0

X_sol = np.hstack([np.vstack(x0) for _ in range(N+1)])
U_sol = np.zeros((nu, N))

for t in range(sim_steps):
    opti.set_value(X0, current_x)
    opti.set_value(X_ref, x_target)

    opti.set_initial(X, X_sol)
    opti.set_initial(U, U_sol)
    
    start_time = time.time()
    # Solve NMPC
    try:
        sol = opti.solve()
    except RuntimeError as e:
        print(f"Solver failed at time {t * dt:.2f}s: {str(e)}")
        break
    solve_times.append(time.time() - start_time)

    X_sol = sol.value(X).reshape(nx, N+1)
    U_sol = sol.value(U).reshape(nu, N)

    u_opt = sol.value(U[0])
    u_hist.append(u_opt)

    # current_x = integrator(x0=current_x, p=u_opt)['xf'].full().flatten()
    current_x = cartpole.sim(current_x, u_opt)
    
    x_hist.append(current_x)
    
    # Prepare warm-start
    X_sol = np.hstack((X_sol[:, 1:], X_sol[:, [-1]]))
    U_sol = np.hstack((U_sol[:, 1:], U_sol[:, [-1]]))

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