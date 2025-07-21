import numpy as np
import json
import math

class Particle:
    # We can initialize a particle with some initial conditions
    def __init__(self, x_i: float, num_bins: int):
        self.x = x_i
        self.num_bins = num_bins
        self.history = []       # Used to store information about the particle's past position

        # Record the initial position
        self.history.append(self.x)

    # Computes which bin boundaries are crossed during an update
    # As well as what direction
    # This MUST be called before a particle's x position is set to be inside the bounds of the simulation,
    # otherwise we wouldn't know which direction it traveled.
    def get_crossings(self, L: float, old_x: float) -> tuple[bool, list]:
        going_right = self.x > old_x

        # Used to ensure that range() is called with the lower number on the bottom
        if going_right:
            rightmost = self.x
            leftmost = old_x
        else:
            rightmost = old_x
            leftmost = self.x

        # By scaling the x positions by num_bins / L, we scale the boundaries of our cells to lie on the integers
        # So now, we can see which boundaries are crossed by simply seeing which integers lie between
        # the x positions before and after the crossing
        # We can do this by ceiling both numbers and using python's range function.
        crossings = range(math.ceil(leftmost / L * self.num_bins), math.ceil(rightmost / L * self.num_bins))

        return going_right, [crossing % self.num_bins for crossing in crossings]


    # Updates the position of the particle for one timestep dt.
    # The coefficient C is sqrt(2 * D * dt) and its computation is handled by the Simulation class
    def update(self, L: float, C: float) -> tuple[bool, list]:
        dx = C * np.random.normal(0, 1)
        old_x = self.x
        self.x += dx

        crossing_data = self.get_crossings(L, old_x)

        # Since we have a periodic boundary condition, we clamp x
        # to between 0 and L using the modulus
        self.x = self.x % L

        # This is how we record the movement of the particle
        self.history.append(self.x)

        # The data about the boundary crossings is given to the parent simulation object, who ends up
        # counting how many crossings happen every update.
        return crossing_data


    # Helper serialization method
    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__)


    # Helper deserialization method
    @staticmethod
    def from_json(json_data: dict) -> 'Particle':
        particle = Particle(json_data['x'], json_data['num_bins'])
        particle.history = json_data['history']

        return particle