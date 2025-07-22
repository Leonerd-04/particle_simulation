import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.animation as animation
from matplotlib import colors
from simulation import Simulation
import numpy as np
from functools import reduce
import scipy.stats as stats
import solutions

# Make a plot of the simulation using matplotlib and show it to the user
# If save_to is specified, the plot is saved to whatever directory is specified by the user.
def plot(simulation: Simulation, save_to: str =None):
    # Parameters to make the plots look a bit nicer
    params = {
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'axes.titleweight': 'bold'
    }
    plt.rcParams.update(params)

    # The time values to plot against
    t_values = [i * simulation.dt for i in range(simulation.current_step)]

    for particle in simulation.particles:
        plt.scatter(t_values, particle.history, s=2)

    plt.xlabel('Time')
    plt.ylabel('Particle position')

    if save_to:
        plt.savefig(save_to)

    plt.show()


# Make a plot of the simulation's histogram using matplotlib and show it to the user
# If save_to is specified, the plot is saved to whatever directory is specified by the user.
# Generates the histograms using the automatic function from matplotlib
# Unused
def plot_hists(simulation: Simulation, save_to: str =None):
    num_x = simulation.histogram_config['num_x']
    num_t = simulation.histogram_config['num_t']
    density = simulation.histogram_config['number_density']

    # Gets us linearly spaced t values to sample
    # Rounded using integer truncation
    # Units are multiples of dt.
    t_values = [int(np.floor(t)) for t in np.linspace(0, simulation.current_step - 1, num_t)]
    # print(t_values)

    fig, ax = plt.subplots()
    artists = []

    # Stores the histograms if they are to be saved
    hists = []
    for t in t_values:
        # The distribution is a snapshot of the particles we want to plot at time t
        distribution = [particle.history[t] for particle in simulation.particles]
        N, bins, patches = ax.hist(distribution, bins=num_x, range=(0, simulation.L), density=True)

        # N represents the values of the histogram bins, so we can use it to set their colors.
        fracs = N / N.max()

        # we need to normalize the data to 0->1 for the full range of the colormap
        if t <= 0:
            norm = colors.Normalize(0, fracs.max())

        # Now, we'll loop through our objects and set the color of each accordingly
        for thisfrac, thispatch in zip(fracs, patches):
            color = plt.cm.viridis(norm(thisfrac))
            thispatch.set_facecolor(color)

        # Add it to our list of artists to animate
        artists.append(patches)

        if save_to:
            hists.append(N)

    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=125)
    plt.show()

# Make a plot of the simulation's histogram using matplotlib and show it to the user
# Uses stairs and pre-generated histograms instead of letting them be generated.
def plot_hists_generated(simulation: Simulation, save_to: str =None):
    hists, edges = simulation.generate_hist()

    # Gets us linearly spaced t values to sample
    # Rounded using integer truncation
    # Units are multiples of dt.
    t_values = [int(np.floor(t)) for t in np.linspace(0, simulation.current_step - 1, simulation.histogram_config['num_t'])]
    x_values = np.linspace(0, simulation.L, 100)

    # Gets only the crossings for the particular values we want
    crossings = [simulation.bin_crossings[t] for t in t_values]

    fig, ax = plt.subplots()
    artists = []
    for hist, crossing, k in zip(hists, crossings, t_values):
        hist_patch = ax.stairs(hist, edges, fill=True)
        hist_patch.set_facecolor((0.80, 0.30, 0.50))

        # Shows the boundary condition better to also have the first crossing at the end
        crossing.append(crossing[0])
        crossing_per_time = [c / simulation.dt for c in crossing]
        stem_patch = ax.stem(edges, crossing_per_time)

        f = simulation.get_fourier_bound_func(10)

        y_values = [f(x, k * simulation.dt) for x in x_values]
        fourier_patch = ax.plot(x_values, y_values, color=(0.05, 0.50, 0.24))

        g = simulation.get_gaussian_bound_func(5)

        y_values = [g(x, k * simulation.dt) for x in x_values]
        gauss_patch = ax.plot(x_values, y_values, color=(0.15, 0.10, 0.40))

        patches = list(stem_patch)
        patches.append(hist_patch)

        for patch in fourier_patch:
            patches.append(patch)

        for patch in gauss_patch:
            patches.append(patch)

        artists.append(patches)

    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=500)
    plt.show()

    # If the user specified a save location, save the histograms to a txt
    if save_to:
        simulation.save_hist_txt(save_to)


# Plots the values of our histograms to see their probability distribution
# Intended to be used with a simulation initialized to a uniform distribution, with number_density = true.
# Takes in an Axes object to be plotted alongside other graphs
def plot_aggregated_hists(sim: Simulation, ax: Axes):
    hists_2D, _ = sim.generate_hist()               # Get our histogram values
    hists = reduce(lambda x, y: x + y, hists_2D)    # Reduce them to one array
    hist = [arr[0] for arr in hists_2D]             # This gets us the statistics for just one bin if needed

    # This comes from my analysis
    n = sim.num_particles / sim.L
    dx = sim.L / sim.histogram_config['num_x']
    expected_mean = n
    expected_var = n / dx * (1 - 1/sim.histogram_config['num_x'])

    empirical_mean = np.mean(hists)
    empirical_var = np.var(hists)

    ax.text(0.97, 0.93, f"Analysis: µ={expected_mean}, σ²={expected_var:.2f}", transform=ax.transAxes, ha='right')
    ax.text(0.97, 0.86, f"Data: µ={empirical_mean}, σ²={empirical_var:.2f}", transform=ax.transAxes, ha= 'right')
    ax.set_xlabel('Number Density')
    ax.set_ylabel('Probability Density')
    ax.set_title(f"N={sim.num_particles}, dx={dx}, D={sim.D}")



    ax.hist(hists, bins=int(np.log10(sim.num_particles) * 6.25), density=True, label="Simulation Data")

    sigma = np.sqrt(expected_var)

    left = expected_mean - 3 * sigma
    right = expected_mean + 3 * sigma

    x = np.linspace(left, right, 1000)
    ax.plot(x, stats.norm.pdf(x, expected_mean, sigma), label="Bell Curve")

    x_ints = [int(np.floor(t * dx)) for t in x]
    y = [y * dx for y in stats.binom.pmf(x_ints, sim.num_particles , 1/sim.histogram_config['num_x'])]
    ax.plot(x, y, label="Binomial PMF")

    ax.legend(loc="upper left")


def plot_flux_hists(sim: Simulation, ax: Axes):
    hists_2D = sim.bin_crossings[1:]               # Get our histogram values
    hists = [N / sim.dt for N in reduce(lambda x, y: x + y, hists_2D)]   # Reduce them to one array
    hist = [arr[0] for arr in hists_2D]             # This gets us the statistics for just one bin if needed

    # This comes from my analysis
    n = sim.num_particles / sim.L
    expected_mean = 0
    expected_var = 2 * n * np.sqrt(sim.D / (np.pi * sim.dt ** 3))

    empirical_mean = np.mean(hists)
    empirical_var = np.var(hists)

    # ax.text(0.01, 0.79, f"N={sim.num_particles}, dt={sim.dt}, D={sim.D}", transform=ax.transAxes, ha= 'left')
    ax.text(0.97, 0.93, f"Analysis: µ={expected_mean}, σ²={expected_var:.2f}", transform=ax.transAxes, ha='right')
    ax.text(0.97, 0.86, f"Data: µ={empirical_mean:.2f}, σ²={empirical_var:.2f}", transform=ax.transAxes, ha= 'right')
    ax.set_xlabel('Flux (number/second)')
    ax.set_ylabel('Probability Density')
    ax.set_title( f"N={sim.num_particles}, D={sim.D}, dx={sim.L/sim.histogram_config['num_x']}, dt={sim.dt}")

    sigma = np.sqrt(expected_var)

    left = expected_mean - 5 * sigma
    right = expected_mean + 5 * sigma

    bin_left = int(left * sim.dt) / sim.dt
    bin_right = int(right * sim.dt) / sim.dt

    bins = np.linspace(bin_left, bin_right, int(right * sim.dt) - int(left * sim.dt))

    ax.hist(hists, bins=bins, density=True, label="Simulation Data")

    x = np.linspace(left, right, 1000)
    ax.plot(x, stats.norm.pdf(x, expected_mean, sigma), label="Bell Curve")

    ax.legend(loc="upper left")


def plot_hist_at_step(simulation: Simulation, t: int, ax: Axes, flux=False):
    ax.set_xlabel('x position')
    ax.set_title( f"t = {t * simulation.dt}")

    hist, edges = simulation.generate_single_hist_at(t)

    crossings = simulation.bin_crossings[t]
    x_values = np.linspace(0, simulation.L, 250)

    hist_patch = ax.stairs(hist, edges, fill=True, label="Number Density (m^-1)")
    hist_patch.set_facecolor((0.80, 0.30, 0.50))

    if flux:
        # Shows the boundary condition better to also have the first crossing at the end
        crossings.append(crossings[0])
        crossing_per_time = [c / simulation.dt for c in crossings]
        ax.stem(edges, crossing_per_time, label="Flux (s^-1)", linefmt="orange")

    if flux:
        f = simulation.get_fourier_bound_func(10)
        g = simulation.get_gaussian_bound_func(5)
    else:
        f = simulation.get_fourier_func(10)
        g = simulation.get_gaussian_func(5)

    y_values = [f(x, t * simulation.dt) for x in x_values]
    ax.plot(x_values, y_values, color=(0.08, 0.60, 0.36), label="Sine", linewidth=2)

    y_values = [g(x, t * simulation.dt) for x in x_values]
    ax.plot(x_values, y_values, color=(0.00, 0.30, 0.45), label="Gauss", linewidth=3, linestyle='dashed')

    ax.legend(loc="upper left", fontsize='small')


def plot_hists_at_steps(simulation: Simulation, t_values: list[int], num_x: int, num_y: int, flux=False, save_to=None):
    fig, axes = plt.subplots(num_y, num_x, squeeze=False, layout='constrained')

    for i in range(len(t_values)):
        simulation.plot_hist_at_step(t_values[i], axes[i % num_y, i // num_y], flux)

    fig.set_size_inches(15, 5)

    if save_to:
        plt.savefig(save_to)

    plt.show()


def plot_multiple_hists(sims: list[Simulation], num_x: int, num_y: int, flux=False):
    fig, axes = plt.subplots(num_x, num_y, squeeze=True, layout='constrained')

    for i in range(len(sims)):
        if flux:
            plot_flux_hists(sims[i], axes[i % 2, i // 2])
        else:
            plot_aggregated_hists(sims[i], axes[i % 2, i // 2])

    plt.show()




# Add these to the simulation class to be able to use the dot notation
Simulation.plot = plot
Simulation.plot_hists = plot_hists
Simulation.plot_hists_generated = plot_hists_generated
Simulation.plot_aggregated_hists = plot_aggregated_hists
Simulation.plot_hist_at_step = plot_hist_at_step
Simulation.plot_hists_at_steps = plot_hists_at_steps