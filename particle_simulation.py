import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import time


class ParticleSimulation:
    def __init__(self, n_particles=1000, n_types=12, width=800, height=800, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'mps')
        self.n_particles = n_particles
        self.n_types = n_types
        self.width = width
        self.height = height

        # Initialize particle properties
        self.X = torch.rand(n_particles, device=self.device) * width
        self.Y = torch.rand(n_particles, device=self.device) * height
        self.Vx = torch.rand(n_particles, device=self.device) - 0.5
        self.Vy = torch.rand(n_particles, device=self.device) - 0.5
        self.Type = torch.randint(0, n_types, (n_particles,), device=self.device)

        # Initialize rules
        self.Rules = torch.rand((n_types, n_types), device=self.device) * 2 - 1
        self.Rules.fill_diagonal_(-0.01)  # SelfForce

        # Constants
        self.ACTION_DISTANCE = 6400
        self.FORCE_SCALAR = 1
        self.FORCE_BIAS = -0.5

    def update(self):
        start = time.time()
        # Compute all pairwise distances
        dx = self.X.unsqueeze(1) - self.X.unsqueeze(0)
        dy = self.Y.unsqueeze(1) - self.Y.unsqueeze(0)
        d_squared = dx ** 2 + dy ** 2

        # Apply force only within action distance
        mask = (d_squared < self.ACTION_DISTANCE) & (d_squared > 0)

        # Compute forces
        f = torch.zeros_like(d_squared)
        f[mask] = 1 / torch.sqrt(d_squared[mask])

        # Apply rules
        rule_matrix = self.Rules[self.Type.unsqueeze(1), self.Type.unsqueeze(0)]
        f *= rule_matrix

        # Compute force components
        fx = torch.sum(f * dx, dim=1)
        fy = torch.sum(f * dy, dim=1)

        # Update velocities
        self.Vx = (self.Vx + fx) / 2
        self.Vy = (self.Vy + fy) / 2

        # Update positions
        self.X += self.Vx
        self.Y += self.Vy

        # Boundary conditions
        self.X = torch.clamp(self.X, 0, self.width)
        self.Y = torch.clamp(self.Y, 0, self.height)

        # Reverse velocity at boundaries
        self.Vx = torch.where(self.X <= 0, -self.Vx, self.Vx)
        self.Vx = torch.where(self.X >= self.width, -self.Vx, self.Vx)
        self.Vy = torch.where(self.Y <= 0, -self.Vy, self.Vy)
        self.Vy = torch.where(self.Y >= self.height, -self.Vy, self.Vy)
        end = time.time()
        print(f"Update step finished in (time.time()): {end - start:.6f} seconds")

    def run_simulation(self, n_steps=1000):
        for _ in range(n_steps):
            self.update()

    def get_particle_data(self):
        return self.X.cpu().numpy(), self.Y.cpu().numpy(), self.Type.cpu().numpy()


# Visualization function
def visualize_simulation(simulation, n_frames=2000):
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='black')
    ax.set_facecolor('black')
    ax.set_xlim(0, simulation.width)
    ax.set_ylim(0, simulation.height)

    # Set axis color to white
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    colors = plt.cm.rainbow(np.linspace(0, 1, simulation.n_types))
    scatter = ax.scatter([], [], s=1)

    def init():
        scatter.set_offsets(np.c_[[], []])
        return scatter,

    def update(frame):
        simulation.update()
        X, Y, Types = simulation.get_particle_data()
        scatter.set_offsets(np.c_[X, Y])
        scatter.set_color(colors[Types])
        return scatter,

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, interval=0, blit=True)
    plt.close(fig)  # Prevent the initial empty plot from showing
    return anim


# Main execution
if __name__ == "__main__":
    simulation = ParticleSimulation()
    anim = visualize_simulation(simulation)

    # Save the animation as a gif
    anim.save('particle_simulation.gif', writer='pillow', fps=30)

    # If you want to display the animation in a notebook or interactive environment, use:
    # plt.show()
