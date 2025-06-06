import numpy as np

class Particle:
    # This coefficient is sqrt(2 * D * dt).
    # Since D and dt generally stay constant, we only compute it once and store the result.
    COEFFICIENT = 1

    # We can initialize a particle with some initial conditions
    def __init__ (self, x_i: float):
        self.x = x_i

    # Computes the coefficient
    # D is the diffusion constant
    # dt is the size of our time step
    @staticmethod
    def set_coefficient(D: float, dt: float) -> None:
        Particle.COEFFICIENT = np.sqrt(2 * D * dt)

    # Updates the position of the particle for one timestep dt.
    def update(self, L: float) -> None:
        dx = self.COEFFICIENT * np.random.normal(0, 1)
        self.x += dx

        # Since we have a periodic boundary condition, we clamp x
        # to between 0 and L using the modulus
        self.x = self.x % L
