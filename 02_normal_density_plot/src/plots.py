import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
from simulation import Simulation
import numpy as np

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
# Surprisingly doesn't require generate_hist()
# Unused
def plot_hists(simulation: Simulation, num_x: int, num_t: int, save_to: str =None):

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
def plot_hists_generated(simulation: Simulation, num_x: int, num_t: int, density=True, save_to: str =None):
    hists, edges = simulation.generate_hist(num_x, num_t)

    fig, ax = plt.subplots()
    artists = []
    for hist in hists:
        patch = ax.stairs(hist, edges, fill=True)
        patch.set_facecolor((0.83, 0.04, 0.30))

        artists.append([patch])

    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=125)
    plt.show()

    # If the user specified a save location, save the histograms to a txt
    if save_to:
        simulation.save_hist_txt(num_x, num_t, save_to, density)


# Add these to the simulation class
Simulation.plot = plot
Simulation.plot_hists = plot_hists
Simulation.plot_hists_generated = plot_hists_generated