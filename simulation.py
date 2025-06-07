from particle import Particle
import numpy as np
import matplotlib.pyplot as plt

# A class representing a simulation
class Simulation:

    # L is the bound of our simulation. It goes from 0 up to L.
    # params contains any miscellaneous parameters, namely D and dt
    def __init__(self, params: dict):
        self.L = params['L']
        self.num_particles = params['num_particles']
        self.dt = params['dt']
        self.D = params['D']
        self.particle_init = params['p_init']
        self.particles = []
        self.time_step = 0

        # This coefficient is the sqrt(2 * D * dt).
        # Since D and dt generally stay constant, we only compute it once and store the result.
        self.coefficient = 1

        self.init_particles()

        self.set_coefficient(self.D, self.dt)


    # Computes the coefficient
    # D is the diffusion constant
    # dt is the size of our time step
    def set_coefficient(self, D: float, dt: float):
        self.coefficient = np.sqrt(2 * D * dt)

    # Initializes all the particles to a random location
    def init_particles(self):
        self.particles = []
        for i in range(self.num_particles):
            # The particles are initialized according to a distribution specified in params.
            # This distribution must go between 0 and 1.
            x_i = self.particle_init() * self.L
            self.particles.append(Particle(x_i))

        # Increments the time step
        self.time_step += 1

    # Runs one step of the simulation
    def step(self):
        for particle in self.particles:
            particle.update(self.L, self.coefficient)
        self.time_step += 1


    # Runs a given number of steps
    def run_steps(self, steps: int):
        for i in range(steps):
            self.step()

    # Runs for the given amount of time
    def run(self, time: float):
        steps = time // self.dt
        self.run_steps(steps)

    def plot(self):
        # Parameters to make the plots look a bit nicer
        params = {
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'axes.titleweight': 'bold'
        }
        plt.rcParams.update(params)

        t_values = [i * self.dt for i in range(self.time_step)]

        for particle in self.particles:
            plt.plot(t_values, particle.history)

        plt.xlabel('Time')
        plt.ylabel('Particle position')
        plt.show()

    # Formats a string to display an output on the console
    def format_string(self) -> str:
        width = 10
        s = ""

        # First row consists of particle labels and the label for time
        s += f"{'Time:' :<{width - 1}}|"

        for i in range(self.num_particles):
            s += f"{f'P{i + 1}' :<{width}}"

        s += f"\n"

        # Subsequent rows list the time value, followed by the x value for each particle at a given point
        for i in range(self.time_step):
            s += f"{f'{i * self.dt:0.2f}' :<{width - 1}}|"
            for particle in self.particles:
                s += f"{f'{particle.history[i]:0.2f}' :<{width}}"
            s += "\n"

        return s

    def print(self):
        print(self)

    def __str__(self):
        return self.format_string()