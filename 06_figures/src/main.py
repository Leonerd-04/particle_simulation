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

    for i in range(4):
        params = load_config(f"../configs/config_{i}.json")
        simulation = Simulation(params)

        simulation.run_steps(100)
        simulation.save_to(f"../../out/sim5_{i}.json")



if __name__ == "__main__":
    main()
    plot_graphs()