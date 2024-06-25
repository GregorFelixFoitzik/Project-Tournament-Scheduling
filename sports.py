# Standard library
import os
# Source: https://stackoverflow.com/a/48665619, accessed 25th June
# Might need adaption if other CPU is used
os.environ["MKL_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import time
import argparse

# Third party libraries
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
    metaheuristic_name = "lns"
    parameters = {
        # "alpha": 0.95,
        # "temperature": 10000,
        # "epsilon": 0.001,
        # "neighborhood": "random_swap_within_week",
    }

    metaheuristic = METAHEURISTICS[metaheuristic_name](
        algo_config=algo_config,
        timeout=30 - (time.time() - t0),
        start_solution=init_sol,
        **parameters,
    )
    new_sol = metaheuristic.run()
    duration = np.round(a=time.time() - t0, decimals=2)
    profit = compute_profit(
        sol=new_sol, profit=metaheuristic.p, weeks_between=int(algo_config["r"])
    )

    solution_valid = validate(sol=new_sol, num_teams=algo_config["n"])
    print(
        compute_profit(
            sol=init_sol, profit=metaheuristic.p, weeks_between=int(algo_config["r"])
        )
    )
    if solution_valid:
        print_feasible_solution(sol=new_sol, runtime=duration, profit=profit)
    else:
        print(f"### RESULT: {timeout}")
