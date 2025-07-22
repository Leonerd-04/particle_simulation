import numpy as np
import plots
from simulation import Simulation
import json
import os
from post_process import *

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
    # If the out directory doesn't exist yet, create it
    try:
        os.mkdir("../../out")
        print("output directory successfully created")
    except FileExistsError:
        print("output directory already exists")
    except Exception as e:
        print(f"An error occurred: {e}")


    for i in range(1, 4):
        config1 = load_config(f"../configs/config1-{i}.json")
        sim1 = Simulation(config1)

        sim1.run_steps(200)

        sim1.save_to(f"../../out/sim6-1-{i}.json", as_json=True)


    # config2 = load_config("../configs/config2.json")
    # config2['p_init'] = lambda: 0.5
    # sim2 = Simulation(config2)
#
    # sim2.run_steps(200)
#
    # sim2.save_to("../../out/sim6-2.json", as_json=True)
#
#
    # config3 = load_config("../configs/config3.json")
    # config3['p_init'] = lambda: 0.5
    # sim3 = Simulation(config3)
#
    # sim3.run_steps(500)
#
    # sim3.save_to("../../out/sim6-3.json", as_json=True)


if __name__ == "__main__":
    main()
    plot_graphs()