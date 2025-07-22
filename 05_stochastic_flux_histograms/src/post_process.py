from simulation import Simulation
import plots

def plot_graphs():
    simulations = []
    for i in range(4):
        simulations.append(Simulation.open(f"../../out/sim5_{i}.json"))

    plots.plot_multiple_hists(simulations, 2, 2, flux=True, save_to=f"../../out/sim5.png")


if __name__ == "__main__":
    plot_graphs()

