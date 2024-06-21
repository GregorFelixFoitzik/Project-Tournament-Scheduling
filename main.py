# Standard library
import os
import argparse

# Project specific library
import yaml

from src.helper import read_in_file, generate_solution_round_robin_tournament
from src.greedy_heuristic import GreedyHeuristic
from src.metaheuristics.metaheuristics_controller import main_metaheuristics_controller

# Copied from https://docs.python.org/3/library/argparse.html, accessed 28.05.2024
parser = argparse.ArgumentParser(description="Input for algorithms")
# ChatGPT fixed the issues that default values were not used. => python main.py works now
parser.add_argument(
    "-t", "--timeout", type=int, default=30, help="Timeout duration in seconds"
)
parser.add_argument(
    "-p",
    "--path_to_instance",
    type=str,
    default="data/example.in",
    help="Path to the input file",
)


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
    main()
    # Extract command line arguments
    # args = parser.parse_args()
    # path_to_file = args.path_to_instance
    # timeout = args.timeout

    # algo_config = read_in_file(path_to_file='instances/dotl_n10_t0.666_s4_r2_mnunif_1234.in')
    # agent = GreedyHeuristic(algo_config)
    # agent.execute_cmd()
#
# print('ads')
