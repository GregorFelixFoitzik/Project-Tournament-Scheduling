# Standard library
from importlib.resources import path
import os
import time
import argparse

# Third party libraries
import yaml
import numpy as np

# Project specific library
from src.validation import validate
from src.helper import compute_profit, print_feasible_solution
from src.helper import read_in_file, generate_solution_round_robin_tournament
from src.metaheuristics.large_neighborhood_search import LNS
from src.metaheuristics.simulated_annealing import SimulatedAnnealing
from src.metaheuristics.large_neighborhood_search_simulated_annealing import (
    LNSSimAnnealing,
)
from src.metaheuristics.metaheuristics_controller import main_metaheuristics_controller

# Copied from https://docs.python.org/3/library/argparse.html, accessed 28.05.2024
parser = argparse.ArgumentParser(description="Input for algorithms")
# ChatGPT fixed the issues that default values were not used. => python main.py works now
parser.add_argument(
    "timeout", type=float, default=30, help="Timeout duration in seconds"
)
parser.add_argument(
    "path_to_instance",
    type=str,
    default="data/example.in",
    help="Path to the input file",
)

METAHEURISTICS = {
    "lns": LNS,
    "simulated_annealing": SimulatedAnnealing,
    "large_neighborhood_search_simulated_annealing": LNSSimAnnealing,
}


def main():
    with open(file="configs/run_config.yaml", mode="r") as file:
        run_config = yaml.safe_load(stream=file)

    file_names = os.listdir(path="instances")
    for file_name in file_names:
        path_to_file = f"instances/{file_name}"

        algo_config = read_in_file(path_to_file=path_to_file)

        metaheuristics_to_use = run_config["metaheuristics"]
        start_sol = generate_solution_round_robin_tournament(
            num_teams=int(algo_config["n"]),
            t=float(algo_config["t"]),
            random_team_order=True,
        )

        main_metaheuristics_controller(
            start_sol=start_sol,
            metaheuristics_to_use=metaheuristics_to_use,
            algo_config=algo_config,
            time_out=4,
        )


if __name__ == "__main__":
    # main()
    # Extract command line arguments
    args = parser.parse_args()
    path_to_file = args.path_to_instance
    timeout = args.timeout

    algo_config = read_in_file(path_to_file=path_to_file)

    t0 = time.time()
    init_sol = generate_solution_round_robin_tournament(
        num_teams=int(algo_config["n"]),
        t=float(algo_config["t"]),
        random_team_order=True,
    )
    metaheuristic_name = "simulated_annealing"
    parameters = {
        "alpha": 0.95,
        "temperature": 10000,
        "epsilon": 0.001,
        "neighborhood": "random_swap_within_week",
    }

    metaheuristic = METAHEURISTICS[metaheuristic_name](
        algo_config=algo_config,
        time_out=timeout - t0,
        start_solution=init_sol,
        **parameters,
    )
    new_sol = metaheuristic.run()
    duration = np.round(a=time.time() - t0, decimals=2)
    profit = compute_profit(
        sol=new_sol, profit=metaheuristic.p, weeks_between=int(algo_config["r"])
    )

    solution_valid = validate(sol=new_sol, num_teams=algo_config["n"])
    if solution_valid:
        print_feasible_solution(sol=new_sol, runtime=time.time() - t0, profit=profit)
    else:
        print(f"### RESULT: {timeout}")
