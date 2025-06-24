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

# Defines and returns the fourier series formulation for the boundary crossings
# n is how far out to truncate it.
def get_fourier_bound_func(sim: Simulation, n: int) -> Callable[[float, float], float]:
    N = sim.num_particles
    L = sim.L
    D = sim.D
    x_0 = sim.initializer() * L        # For now, just evaluate the initializer to get x_0.
    m = 2 * np.pi / L               # To avoid recomputing this expression

    def fourier(x: float, t: float) -> float:
        sum = 0
        for i in range(1, n + 1):
            w = m * i
            sum += w * np.exp(-D * w ** 2 * t) * np.sin(w * (x - x_0))

        sum *= 2
        sum *= N / L

        return D * sum

    return fourier


# Defines and returns the other series function with gaussians
# n is how far out to truncate it.
def get_gaussian_func(sim: Simulation, n: int) -> Callable[[float, float], float]:
    N = sim.num_particles
    L = sim.L
    D = sim.D
    x_0 = sim.initializer() * L        # For now, just evaluate the initializer to get x_0.

    def gaussian(x: float, t: float) -> float:
        up_to = np.sqrt(2 * D * t) // L * 3 # This ensures we always include 3 standard deviations
        # n truncates how many gaussians we compute
        sum = np.exp(-(x - x_0 )**2 / (4 * D * t))
        for m in range(1, min(int(up_to), n)):
            sum += np.exp(-(x - x_0 + m * L)**2 / (4 * D * t))
            sum += np.exp(-(x - x_0 - m * L)**2 / (4 * D * t))

        sum *= N / (2 * np.sqrt(np.pi * D * t))

        return sum

    return gaussian

# Defines and returns the gaussian formulation of the boundary crossings
# n is how far out to truncate it.
def get_gaussian_bound_func(sim: Simulation, n: int) -> Callable[[float, float], float]:
    N = sim.num_particles
    L = sim.L
    D = sim.D
    x_0 = sim.initializer() * L        # For now, just evaluate the initializer to get x_0.

    def gaussian(x: float, t: float) -> float:
        up_to = np.sqrt(2 * D * t) // L * 3 # This ensures we always include 3 standard deviations
        # n truncates how many gaussians we compute
        sum = 2 * (x - x_0 ) / (4 * D * t) * np.exp(-(x - x_0 )**2 / (4 * D * t))
        for m in range(1, min(int(up_to), n)):
            sum += (x - x_0 + m * L) / (2 * D * t) * np.exp(-(x - x_0 + m * L)**2 / (4 * D * t))
            sum += (x - x_0 - m * L) / (2 * D * t) * np.exp(-(x - x_0 - m * L)**2 / (4 * D * t))

        sum *= N / (2 * np.sqrt(np.pi * D * t))

        return D * sum

    return gaussian



Simulation.get_fourier_func = get_fourier_func
Simulation.get_gaussian_func = get_gaussian_func
Simulation.get_fourier_bound_func = get_fourier_bound_func
Simulation.get_gaussian_bound_func = get_gaussian_bound_func