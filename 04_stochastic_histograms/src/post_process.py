from simulation import Simulation
import plots

def plot_graphs():
    simulations = []
    for i in range(4):
        simulations.append(Simulation.open(f"../../out/sim4_{i}.json"))

    plots.plot_multiple_hists(simulations, 2, 2, save_to=f"../../out/sim4.png")


if __name__ == "__main__":
    plot_graphs()


