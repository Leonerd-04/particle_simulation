import numpy as np
import plots
from simulation import Simulation
import json
import os

config_path = "../configs/config_0.json"
out_path = "../../out/sim3"


def load_config(path: str) -> dict:
    with open(path, 'r') as file:
        json_string = file.read()

    return json.loads(json_string)


def save_config(config: dict, path: str):
    with open(path, 'w') as file:
        json_string = json.dumps(config)
        file.write(json_string)


def main():
    # The parameters of our simulation
    params = load_config(config_path)

    # params['p_init'] = lambda: np.random.normal(0.5, 0.01)
    # params['p_init'] = lambda: 0.5
    simulation = Simulation(params)

    simulation.run_steps(100)
    # simulation.print()
    # simulation.plot()

    # If the out directory doesn't exist yet, create it
    try:
        os.mkdir("../../out")
        print("output directory successfully created")
    except FileExistsError:
        print("output directory already exists")
    except Exception as e:
        print(f"An error occurred: {e}")

    simulation.save_to(f"{out_path}.json")
    simulation.save_to(f"{out_path}.txt", as_json=False)


    plots.plot_multiple_hists([simulation, simulation])



if __name__ == "__main__":
    main()