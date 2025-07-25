import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
from simulation import Simulation
import numpy as np
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


# Add these to the simulation class to be able to use the dot notation
Simulation.plot = plot
Simulation.plot_hists = plot_hists
Simulation.plot_hists_generated = plot_hists_generated