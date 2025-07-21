from collections.abc import Callable
from particle import Particle
import numpy as np
import json

# A class representing a simulation
class Simulation:

    # L is the bound of our simulation. It goes from 0 up to L.
    # params contains any miscellaneous parameters, namely D and dt, as well as the parameters to track histograms
    def __init__(self, params: dict):
        self.L = params['L']
        self.num_particles = params['num_particles']
        self.dt = params['dt']
        self.D = params['D']

        self.histogram_config = params['histogram_config']

        self.particles = []
        self.bin_crossings = []
        self.current_step = 1       # Represents the number of steps that have been computed,
                                    # including the initial step
        if 'bin_crossings' in params.keys():
            self.bin_crossings = params['bin_crossings']
        else:
            # Append a "0th" bin crossing to align this array with the histograms
            self.bin_crossings.append([0] * self.histogram_config['num_x'])

        # This coefficient multiplies with the normal distribution to step the simulation forward.
        # I assume D and dt will stay constant throughout the simulation
        # So we only have to compute it once.
        self.coefficient = np.sqrt(2 * self.D * self.dt)

        # Currently, the initializer function is left unspecified due to how the options
        # are read from a json. This deals with its default value.
        if 'p_init' in params.keys():
            self.initializer = params['p_init']
        else:
            self.initializer = np.random.random

        self.init_particles(self.initializer)


    # Initializes all the particles to a random location in bounds.
    # The particles are initialized according to a distribution specified by initializer
    # This distribution must range from 0 to 1.
    def init_particles(self, initializer: Callable[[], float]):
        self.particles = []
        for i in range(self.num_particles):
            x_i = initializer() * self.L
            self.particles.append(Particle(x_i, self.histogram_config['num_x']))


    # Runs one step of the simulation
    # Also calculates the number of particles that cross cells in this step
    def step(self):
        crossings = [0] * self.histogram_config['num_x']
        for particle in self.particles:
            is_going_right, crossed_bins = particle.update(self.L, self.coefficient)

            if is_going_right:
                increment = 1
            else:
                increment = -1

            for boundary in crossed_bins:
                crossings[boundary] += increment

        self.current_step += 1
        self.bin_crossings.append(crossings)


    # Runs a given number of steps
    def run_steps(self, steps: int):
        for i in range(steps):
            self.step()


    # Runs for the given amount of time
    def run(self, time: float):
        steps = time // self.dt
        self.run_steps(steps)


    # Samples the simulation's history num_t times to get num_t histograms
    # each with num_x cells
    # Returns the histograms themselves and the edges of the bins, for graphing purposes.
    def generate_hist(self) -> tuple[list[list], list]:
        num_x = self.histogram_config['num_x']
        num_t = self.histogram_config['num_t']
        density = self.histogram_config['number_density']

        dx = self.L / num_x

        # Gets us linearly spaced t values to sample
        # Rounded using integer truncation
        # Units are multiples of dt.
        t_values = [int(np.floor(t)) for t in np.linspace(0, self.current_step - 1, num_t)]

        result = []
        for i in t_values:
            # Initializes an array of 0's to represent our cells
            histogram = [0] * num_x
            for particle in self.particles:
                # Integer division by the size of the cell gets us which cell the particle is in
                cell = int(particle.history[i] // dx)

                # There is a particle in the cell just calculated. This counts it.
                histogram[cell] += 1

            if density:
                number_density_histogram = [value / dx for value in histogram]
                result.append(number_density_histogram)
            else:
                result.append(histogram)

        return result, [i * dx for i in range(num_x + 1)]


    def get_hist_txt(self) -> str:
        hists, _ = self.generate_hist()
        num_t = self.histogram_config['num_t']

        t_values = [int(np.floor(t)) for t in np.linspace(0, self.current_step - 1, num_t)]

        s = ""
        for hist, t in zip(hists, t_values):
            s += f"{self.dt * t} "
            for cell in hist:
                s += f"{cell} "
            s += "\n"

        return s


    def save_hist_txt(self, path: str):
        with open(path, 'w') as file:
            file.write(self.get_hist_txt())

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

