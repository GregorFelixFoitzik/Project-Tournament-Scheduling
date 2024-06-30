# main.py
import os
import sys
import yaml
import csv
import time
import numpy as np
from typing import Union
from src.metaheuristics.reactive_tabu_search import ReactiveTabuSearch
from src.helper import (
    compute_profit,
    generate_solution_round_robin_tournament,
    read_in_file,
)
from src.metaheuristics.lns_ts import LNSTS
from src.metaheuristics.tabu_search import TabuSearch
from src.metaheuristics.large_neighborhood_search import LNS
from src.metaheuristics.simulated_annealing import SimulatedAnnealing
from src.metaheuristics.lns_ts_simulated_annealing import LNSTSSimAnnealing
from src.metaheuristics.variable_neighborhood_search import VNS
from src.metaheuristics.large_neighborhood_search_simulated_annealing import (
    LNSSimAnnealing,
)

METAHEURISTICS = {
    "lns": LNS,
    "simulated_annealing": SimulatedAnnealing,
    "large_neighborhood_search_simulated_annealing": LNSSimAnnealing,
    "tabu_search": TabuSearch,
    "tabu_search_lns": LNSTS,
    "lns_ts_simulated_annealing": LNSTSSimAnnealing,
    "reactive_tabu_search": ReactiveTabuSearch,
    # "vns": VNS,
}


def get_yaml_config(mh_config_file: str) -> dict[str, Union[int, float]]:
    """Load the yaml-config file for a given metaheuristic.

    Args:
        mh_config_file (str): Path to the YAML configuration file.

    Returns:
        dict[str, Union[int, float]]: Dictionary containing the configuration.
    """
    with open(file=mh_config_file, mode="r") as file:
        config = yaml.safe_load(stream=file)
    return config["parameters"]


def apply_metaheuristic(
    metaheuristic_name: str,
    algo_config: dict[str, Union[int, float, np.ndarray]],
    timeout: float,
    parameters: dict[str, Union[int, float]],
) -> list[Union[str, float]]:
    """Apply a metaheuristic to a given problem.

    Args:
        metaheuristic_name (str): Name of the metaheuristic.
        algo_config (dict[str, Union[int, float, np.ndarray]]): Configuration of the algorithm.
        timeout (float): Timeout for the Metaheuristic.
        parameters (dict[str, Union[int, float]]): Parameter-configuration for the Metaheuristic.

    Returns:
        list[Union[str, float]]: List containing the name, profit and duration.
    """
    start_time = time.process_time()  # Start time measurement
    start_sol, rc = generate_solution_round_robin_tournament(
        num_teams=int(algo_config["n"]),
        t=float(algo_config["t"]),
        random_team_order=False,
    )
    metaheuristic = METAHEURISTICS[metaheuristic_name](
        algo_config=algo_config,
        timeout=timeout,
        start_solution=start_sol,
        rc=rc,
        **parameters,
    )
    new_sol = metaheuristic.run()
    end_time = time.process_time()  # End time measurement
    duration = end_time - start_time

    profit_start = compute_profit(
        sol=start_sol,
        profit=np.array(object=list(algo_config["p"])).reshape(
            (3, algo_config["n"], algo_config["n"])
        ),
        weeks_between=int(algo_config["r"]),
    )
    profit_meta = compute_profit(
        sol=new_sol, profit=metaheuristic.p, weeks_between=int(algo_config["r"])
    )

    return [metaheuristic_name, profit_start, profit_meta, duration]


def main():
    if len(sys.argv) != 5:
        print(
            "Usage: python main.py <path_to_instance> <metaheuristic_name> <path_to_config> <timeout>"
        )
        sys.exit(1)

    path_to_instance = sys.argv[1]
    metaheuristic_name = sys.argv[2]
    path_to_config = sys.argv[3]
    timeout = float(sys.argv[4])

    algo_config = read_in_file(path_to_file=path_to_instance)
    parameters = get_yaml_config(mh_config_file=f"configs/{path_to_config}.yaml")

    results = apply_metaheuristic(
        metaheuristic_name=metaheuristic_name,
        algo_config=algo_config,
        timeout=timeout,  # Assuming a default timeout of 30 seconds
        parameters=parameters,
    )

    # Ensure the CSV file exists and append results
    if not os.path.exists("artifacts/evaluations_first_run.csv"):
        with open("artifacts/evaluations_first_run.csv", "w") as f:
            pass
    row = [
        metaheuristic_name,
        algo_config,
        timeout,  # Assuming a default timeout of 30 seconds
        parameters,
        results[1],  # profit_start
        results[2],  # profit_meta
        results[3],  # duration
    ]
    with open("artifacts/evaluations_first_run.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(row)


if __name__ == "__main__":
    main()
