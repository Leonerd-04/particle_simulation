import numpy as np

class Particle:
    # We can initialize a particle with some initial conditions
    def __init__ (self, x_i: float):
        self.x = x_i
        self.history = []

        # Record the initial position
        self.history.append(self.x)


    # Updates the position of the particle for one timestep dt.
    def update(self, L: float, C: float) -> None:
        dx = C * np.random.normal(0, 1)
        self.x += dx

        # Since we have a periodic boundary condition, we clamp x
        # to between 0 and L using the modulus
        self.x = self.x % L

        # This is how we record the movement of the particle
        self.history.append(self.x)
