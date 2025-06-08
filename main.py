from particle import Particle
from simulation import Simulation
import numpy as np
import json

config_path = "config.json"

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

    simulation = Simulation(params)
    simulation.run_steps(1000)
    # simulation.print()
    # simulation.plot()
    j = simulation.to_json()
    s = Simulation.from_json(j)

    s.plot()
    s.print()

    save_config(params, config_path)


if __name__ == "__main__":
    main()