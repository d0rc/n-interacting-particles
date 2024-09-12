# Particle Simulation

A GPU-accelerated particle simulation system implemented in Python using PyTorch and visualized with Matplotlib.

![Particle Simulation Demo](particle_simulation.gif)

## Description

This project implements a multi-particle system simulation where particles of different types interact based on customizable rules. The simulation leverages GPU acceleration for efficient computation, allowing for the simulation of large numbers of particles in real-time.

## Features

- GPU-accelerated particle physics simulation
- Customizable particle types and interaction rules
- Real-time visualization using Matplotlib
- Ability to save simulations as GIF animations

## Requirements

- Python 3.11+
- PyTorch
- NumPy
- Matplotlib

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/d0rc/n-interacting-particles.git
   cd n-interacting-particles
   ```

2. Install the required packages:
   ```
   pip install torch numpy matplotlib
   ```

## Usage

Run the simulation with default parameters:

```python
python particle_simulation.py
```

This will generate a `particle_simulation.gif` file in the current directory.

## Customization

You can customize the simulation by modifying the parameters in the `ParticleSimulation` class:

- `n_particles`: Number of particles in the simulation
- `n_types`: Number of different particle types
- `width` and `height`: Dimensions of the simulation space
- `ACTION_DISTANCE`: Maximum distance for particle interactions
- `FORCE_SCALAR` and `FORCE_BIAS`: Parameters affecting force calculations

## How It Works

1. **Initialization**: Particles are created with random positions, velocities, and types.
2. **Interaction Rules**: A matrix defines how different types of particles interact (attract or repel).
3. **Physics Simulation**: In each step, forces between particles are calculated based on distances and rules.
4. **Movement**: Particle velocities and positions are updated based on the calculated forces.
5. **Boundaries**: Particles bounce off the edges of the simulation space.
6. **Visualization**: The state of the system is rendered as an animation, with particles colored by type.

## Performance

The simulation uses PyTorch tensors and GPU acceleration (if available) for efficient computation of particle interactions. This allows for simulating thousands of particles in real-time.

## Contributing

Contributions to improve the simulation or add new features are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by various particle life simulations and GPU computing techniques.
- Thanks to the PyTorch and Matplotlib communities for their excellent libraries.
