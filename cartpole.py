import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Cartpole:
    def __init__(self, dt):
        self.dt = dt    # Time step
        self.nx = 4     # state: [x, x_dot, theta, theta_dot]
        self.nu = 1     # input: force
        self.m = 0.2    # Pole mass
        self.M = 0.5    # Cart mass
        self.L = 0.3    # Pole length
        self.g = 9.81   # Gravity
        
        self.sim_dt = 0.1 * dt
        
        x = ca.SX.sym('x', self.nx)
        u = ca.SX.sym('u', self.nu)

        xdot = self.dynamics(x, u)

        dae = {'x': x, 'p': u, 'ode': xdot}
        integrator_opts = {
            'tf': self.sim_dt,           # Integration dt
            'simplify': True,
            'number_of_finite_elements': 4  # 4 steps = RK4
        }
        self.integrator = ca.integrator('integrator', 'rk', dae, integrator_opts)

    def dynamics(self, x, u):
        _, x_dot, theta, theta_dot = x[0], x[1], x[2], x[3]
        force = u[0]

        sin_theta = ca.sin(theta)
        cos_theta = ca.cos(theta)
        total_mass = self.m + self.M
        temp = (force + self.m * self.L * theta_dot**2 * sin_theta) / total_mass
        
        theta_ddot = (self.g * sin_theta - cos_theta * temp) / (self.L * (4/3 - self.m * cos_theta**2 / total_mass))
        x_ddot = temp - self.m * self.L * theta_ddot * cos_theta / total_mass
        
        return ca.vertcat(x_dot, x_ddot, theta_dot, theta_ddot)
    
    def sim(self, x, u):
        for _ in range(10):
            x = self.integrator(x0=x, p=u)['xf'].full().flatten()
        return x

    def plot(self, x_hist, u_hist):
        time = np.arange(len(x_hist)) * self.dt

        plt.figure(figsize=(12, 8))
        labels = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]
        for i in range(self.nx):
            plt.subplot(self.nx+1, 1, i+1)
            plt.plot(time, x_hist[:, i])
            plt.title(labels[i])
            plt.grid()

        plt.subplot(self.nx+1, 1, self.nx+1)
        plt.plot(time[:-1], u_hist[:, 0], label='Control Force')
        plt.title("Control Input")
        plt.xlabel("Time [s]")
        plt.grid()

        plt.tight_layout()
        plt.show()

    def animate(self, x_hist):
        cart_w = 0.3
        cart_h = 0.2
        pole_l = self.L

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_xlim(-2, 2)
        ax.set_ylim(-0.5, 1.2)
        ax.set_aspect('equal')
        ax.grid()

        # Elements: cart, pole, axle, trail
        cart_patch = plt.Rectangle((0, 0), cart_w, cart_h, fc='k')
        pole_line, = ax.plot([], [], lw=4, color='blue')

        ax.add_patch(cart_patch)

        def init():
            cart_patch.set_xy((-10, -10))
            pole_line.set_data([], [])
            return cart_patch, pole_line

        def update(frame):
            state = x_hist[frame]
            x = state[0]
            theta = np.pi - state[2]

            # Cart coordinates
            cart_x = x - cart_w / 2
            cart_y = 0

            # Pole coordinates
            axle_x = x
            axle_y = cart_y + cart_h
            tip_x = axle_x + pole_l * np.sin(theta)
            tip_y = axle_y - pole_l * np.cos(theta)

            # Update plot elements
            cart_patch.set_xy((cart_x, cart_y))
            pole_line.set_data([axle_x, tip_x], [axle_y, tip_y])

            return cart_patch, pole_line

        ani = animation.FuncAnimation(fig, update, frames=len(x_hist),
                                init_func=init, blit=True, interval=self.dt*1000)
        plt.show()