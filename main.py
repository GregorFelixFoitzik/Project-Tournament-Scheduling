# Standard library
import os

# Source: https://stackoverflow.com/a/48665619, accessed 25th June
# Might need adaption if other CPU is used
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import time
import argparse
import itertools

# Third party libraries
import numpy as np
import pandas as pd

# Project specific library
import yaml

from src.helper import (
    read_in_file,
    generate_solution_round_robin_tournament,
    compute_profit,
)
from src.metaheuristics.metaheuristics_controller import (
    apply_metaheuristic,
    main_metaheuristics_controller,
)

# Copied from https://docs.python.org/3/library/argparse.html, accessed 28.05.2024
parser = argparse.ArgumentParser(description="Input for algorithms")
# ChatGPT fixed the issues that default values were not used. => python main.py works now
parser.add_argument(
    "path_to_instance",
    type=str,
    default="data/example.in",
    help="Path to the input file",
)

parser.add_argument(
    "metaheuristic_name",
    type=str,
    help="Name of the Metaheuristc.",
)


def grid_search():
    mh_configs = {
        "large_neighborhood_search_simulated_annealing": {
            "alpha": [0.8, 0.85, 0.9, 0.95],
            "temperature": [1000, 2500, 7500, 10000],
            "epsilon": [0.1, 0.01, 0.001, 0.0001],
        },
        "simulated_annealing": {
            "alpha": [0.8, 0.85, 0.9, 0.95],
            "temperature": [1000, 2500, 7500, 10000],
            "epsilon": [0.1, 0.01, 0.001, 0.0001],
            "neighborhood": ["random_swap_within_week", "select_worst_n_weeks"],
        },
        "tabu_search": {"max_size_tabu_list": [i * 10 for i in range(1, 4)]},
    }
    file_names = os.listdir(path="data")
    for file_name in file_names:
        path_to_file = f"data/{file_name}"
        print(f"File name: {path_to_file}")

        algo_config = read_in_file(path_to_file=path_to_file)
        for metaheuristic, configs in mh_configs.items():
            df_res = pd.DataFrame(
                columns=["Metaheuristic", "profit", "duration"] + list(configs.keys())
            )
            print(f"\tApply Grid-Search for {metaheuristic}")

            for combination in list(itertools.product(*tuple(configs.values()))):
                print(f"\t\tCombination {list(configs.keys())}: {combination}")
                t0 = time.time()
                start_sol = generate_solution_round_robin_tournament(
                    num_teams=int(algo_config["n"]),
                    t=float(algo_config["t"]),
                    random_team_order=True,
                )
                results = apply_metaheuristic(
                    metaheuristic_name=metaheuristic,
                    start_sol=start_sol,
                    algo_config=algo_config,
                    timeout=30 - (time.time() - t0),
                    parameters={
                        list(configs.keys())[i]: combination[i]
                        for i in range(len(configs.keys()))
                    },
                )

                profit_start = compute_profit(
                    sol=start_sol,
                    profit=np.array(object=list(algo_config["p"])).reshape(
                        (3, algo_config["n"], algo_config["n"])
                    ),
                    weeks_between=int(algo_config["r"]),
                )
                df_res.loc[len(df_res)] = ["Start sol", profit_start, 0] + list(
                    combination
                )
                df_res.loc[len(df_res)] = results + list(combination)
                print("\t\t" + str(df_res).replace("\n", "\n\t\t"))
                print()
            print()


def main(path_to_file: str, metaheuristic_name: str):
    algo_config = read_in_file(path_to_file=path_to_file)

    start_sol = generate_solution_round_robin_tournament(
        num_teams=int(algo_config["n"]),
        t=float(algo_config["t"]),
        random_team_order=False,
    )

    main_metaheuristics_controller(
        start_sol=start_sol,
        metaheuristic_name=metaheuristic_name,
        algo_config=algo_config,
        timeout=30,
    )


if __name__ == "__main__":
    # grid_search()
    args = parser.parse_args()

    path_to_file = args.path_to_instance
    metaheuristic_name = args.metaheuristic_name
    main(path_to_file=path_to_file, metaheuristic_name=metaheuristic_name)
