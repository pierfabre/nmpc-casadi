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
Example parametric NLP:

    opti = casadi.Opti();

    x = opti.variable(2,1);
    p = opti.parameter();

    opti.minimize(  (p*x(2)-x(1)^2)^2   );
    opti.subject_to( 1<=sum(x)<=2 );

    opti.solver('ipopt');

    opti.set_value(p, 3);
    sol = opti.solve();
    sol.value(x).reshape(2,1)

    opti.set_value(p, 5);
    sol = opti.solve();
    sol.value(x).reshape(2,1)
"""

opti = ca.Opti()

# Decision variables
X = # TODO
U = # TODO

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

    # Running cost
    cost += # TODO

    # Integrate
    x_next = integrator(x0=x_k, p=u_k)['xf']
    
    # Dynamics constraint
    # TODO

# Terminal cost
cost += # TODO

opti.minimize(cost)

# ========== Solver ==========
opti.solver("ipopt")

# ========== Simulation ==========
x_hist = [x0]
u_hist = []
solve_times = []

current_x = x0

# Initialize the initial guess
X_sol = # TODO
U_sol = # TODO

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
    X_sol = # TODO
    U_sol = # TODO

    # Send the action to the cartpole
    u = # TODO
    current_x = cartpole.sim(current_x, u)
    
    u_hist.append(u)
    x_hist.append(current_x)
    
    # Prepare warm-start (shift and extend)
    """Hint: use np.hstack"""
    X_sol = # TODO
    U_sol = # TODO

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