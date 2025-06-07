from particle import Particle
from simulation import Simulation
import numpy as np


def main():
    # The parameters of our simulation
    # Eventually these will be read from a json
    params = {
        "L":    10,                 # The bound size of our simulation; will be [0, L)
        "num_particles": 10,
        "dt": 0.01,
        "D":     5,                 # Diffusion coefficient
        "p_init": np.random.random  # Initial position function
    }

    simulation = Simulation(params)

    simulation.run_steps(100)

    simulation.print()

if __name__ == "__main__":
    main()