import numpy as np
import json

class Particle:
    # We can initialize a particle with some initial conditions
    def __init__(self, x_i: float):
        self.x = x_i
        self.history = []       # Used to store information about the particle's past position

        # Record the initial position
        self.history.append(self.x)


    # Updates the position of the particle for one timestep dt.
    # The coefficient C is sqrt(2 * D * dt) and its computation is handled by the
    # Simulation class
    def update(self, L: float, C: float) -> None:
        dx = C * np.random.normal(0, 1)
        self.x += dx

        # Since we have a periodic boundary condition, we clamp x
        # to between 0 and L using the modulus
        self.x = self.x % L

        # This is how we record the movement of the particle
        self.history.append(self.x)


    # Helper serialization method
    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__)


    # Helper deserialization method
    @staticmethod
    def from_json(json_data: dict) -> 'Particle':
        particle = Particle(json_data['x'])
        particle.history = json_data['history']

        return particle