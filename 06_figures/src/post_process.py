from simulation import Simulation
from matplotlib.pyplot import figure
from plots import *

def plot_graphs():
    sims = []
    for i in range(1, 4):
        sim1 = Simulation.open(f"../../out/sim6-1-{i}.json")
        sims.append(sim1)

    plot_multiple(sims, 3, 1, save_to="../../out/sim6-1.png")

    sim2 = Simulation.open("../../out/sim6-2.json")
    sim2.initializer = lambda: 0.5
    sim2.plot_hists_at_steps([1, 15, 192], 3, 1, save_to="../../out/sim6-2.png")

    sim3 = Simulation.open("../../out/sim6-3.json")
    sim3.initializer = lambda: 0.5
    sim3.plot_hists_at_steps([1, 15, 460], 3, 1, flux=True, save_to="../../out/sim6-3.png")


if __name__ == "__main__":
    plot_graphs()

