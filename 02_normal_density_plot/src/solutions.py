from collections.abc import Callable
import numpy as np
from simulation import Simulation


# Defines and returns the fourier series function
# n is how far out to truncate it.
def get_fourier_func(sim: Simulation, n: int) -> Callable[[float, float], float]:
    N = sim.num_particles
    L = sim.L
    D = sim.D
    x_0 = sim.initializer() * L        # For now, just evaluate the initializer to get x_0.
    m = 2 * np.pi / L               # To avoid recomputing this expression

    def fourier(x: float, t: float) -> float:
        sum = 0
        for i in range(1, n + 1):
            w = m * i
            sum += np.exp(-D * w ** 2 * t) * np.cos(w * (x - x_0))

        sum *= 2
        sum += 1
        sum *= N / L

        return sum

    return fourier


Simulation.get_fourier_func = get_fourier_func