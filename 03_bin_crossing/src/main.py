import numpy as np
import plots
from simulation import Simulation
import json
import os

config_path = "../config.json"
out_path = "../../out/sim2"


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

    params['p_init'] = lambda: np.random.normal(0.5, 0.01)
    simulation = Simulation(params)

    simulation.run_steps(600)
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


    if not os.path.exists(f"{out_path}.json"):
        simulation.run_steps(600)

        simulation.save_to(f"{out_path}.json")
        simulation.save_to(f"{out_path}.txt", as_json=False)


    s = Simulation.open(f"{out_path}.json")

    s.plot_hists_generated(save_to=f"{out_path}_hists.txt")
    # s.print()

    # save_config(params, config_path)


if __name__ == "__main__":
    main()