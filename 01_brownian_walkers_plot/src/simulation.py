from collections.abc import Callable
from particle import Particle
import numpy as np
import matplotlib.pyplot as plt
import json

# A class representing a simulation
class Simulation:

    # L is the bound of our simulation. It goes from 0 up to L.
    # params contains any miscellaneous parameters, namely D and dt
    def __init__(self, params: dict):
        self.L = params['L']
        self.num_particles = params['num_particles']
        self.dt = params['dt']
        self.D = params['D']
        self.particles = []
        self.current_step = 1       # Represents the number of steps that have been computed,
                                    # including the initial step

        # This coefficient multiplies with the normal distribution to step the simulation forward.
        # I assume D and dt will stay constant throughout the simulation
        # So we only have to compute it once.
        self.coefficient = np.sqrt(2 * self.D * self.dt)

        # Currently, the initializer function is left unspecified due to how the options
        # are read from a json. This deals with its default value.
        if 'p_init' in params.keys():
            initializer = params['p_init']
        else:
            initializer = np.random.random

        self.init_particles(initializer)


    # Initializes all the particles to a random location in bounds.
    # The particles are initialized according to a distribution specified by initializer
    # This distribution must range from 0 to 1.
    def init_particles(self, initializer: Callable[[], float]):
        self.particles = []
        for i in range(self.num_particles):
            x_i = initializer() * self.L
            self.particles.append(Particle(x_i))


    # Runs one step of the simulation
    def step(self):
        for particle in self.particles:
            particle.update(self.L, self.coefficient)
        self.current_step += 1


    # Runs a given number of steps
    def run_steps(self, steps: int):
        for i in range(steps):
            self.step()


    # Runs for the given amount of time
    def run(self, time: float):
        steps = time // self.dt
        self.run_steps(steps)


    # Make a plot of the simulation using matplotlib and show it to the user
    # If save_to is specified, the plot is saved to whatever directory is specified by the user.
    def plot(self, save_to: str =None):
        # Parameters to make the plots look a bit nicer
        params = {
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'axes.titleweight': 'bold'
        }
        plt.rcParams.update(params)

        # The time values to plot against
        t_values = [i * self.dt for i in range(self.current_step)]

        for particle in self.particles:
            plt.scatter(t_values, particle.history, s=2)

        plt.xlabel('Time')
        plt.ylabel('Particle position')

        if save_to:
            plt.savefig(save_to)

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
        for i in range(self.current_step):
            s += f"{f'{i * self.dt:0.2f}' :<{width - 1}}|"
            for particle in self.particles:
                s += f"{f'{particle.history[i]:0.2f}' :<{width}}"
            s += "\n"

        return s


    def print(self):
        print(self)


    # Helper serialization method
    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__)


    # Helper deserialization method
    @staticmethod
    def from_json(j: str) -> 'Simulation':
        json_data = json.loads(j)

        simulation = Simulation(json_data)
        simulation.current_step = json_data['current_step']
        simulation.particles = [Particle.from_json(data) for data in json_data['particles']]

        return simulation

    # Returns a simpler format to save to a txt file
    # As requested by Dr. Kim
    def to_txt(self) -> str:

        # The time values to plot against
        t_values = [i * self.dt for i in range(self.current_step)]

        s = ""

        for i in range(self.current_step):
            s += f"{t_values[i]} "
            for particle in self.particles:
                s += f"{particle.history[i]} "
            s += "\n"

        return s

    # Saves a simulation to a given file path
    def save_to(self, path: str, as_json: bool =True):
        with open(path, 'w') as file:
            if as_json:
                file.write(self.to_json())
            else:
                file.write(self.to_txt())


    # Creates a simulation class using the given file
    @staticmethod
    def open(path: str) -> 'Simulation':
        with open(path, 'r') as file:
            json_string = file.read()

        return Simulation.from_json(json_string)


    def __str__(self):
        return self.format_string()

