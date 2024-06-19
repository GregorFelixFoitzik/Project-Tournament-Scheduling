# Standard library
import time

from typing import Union

# Third party libraries
import yaml
import numpy as np
import pandas as pd

# Project specific library
from src.metaheuristics.genetic_algorithms import GeneticAlgorithm
from src.helper import compute_profit, print_solution
from src.metaheuristics.large_neighborhood_search import LNS
from src.metaheuristics.simulated_annealing import SimulatedAnnealing
from src.metaheuristics.large_neighborhood_search_simulated_annealing import (
    LNSSimAnnealing,
)

METAHEURISTICS = {
    "lns": LNS,
    "simulated_annealing": SimulatedAnnealing,
    "large_neighborhood_search_simulated_annealing": LNSSimAnnealing,
    'genetic_algorithm': GeneticAlgorithm
}


def get_yaml_config(metaheuristic_name: str) -> dict[str, Union[int, float]]:
    with open(file=f"configs/{metaheuristic_name}.yaml", mode="r") as file:
        config = yaml.safe_load(stream=file)

    return config["parameters"]


def apply_metaheuristic(
    metaheuristic_name: str,
    start_sol: np.ndarray,
    algo_config: dict[str, Union[int, float, np.ndarray]],
    time_out: int,
    parameters: dict[str, Union[int, float]],
) -> list[Union[str, float]]:
    if metaheuristic_name == 'genetic_algorithm':
        metaheuristic = METAHEURISTICS[metaheuristic_name](
            algo_config=algo_config,
            time_out=time_out,
            **parameters,
        )
    else:
        metaheuristic = METAHEURISTICS[metaheuristic_name](
            algo_config=algo_config,
            time_out=time_out,
            start_solution=start_sol,
            **parameters,
        )
    t0 = time.time()
    new_sol = metaheuristic.run()
    duration = np.round(a=time.time() - t0, decimals=2)
    profit = compute_profit(
        sol=new_sol, profit=metaheuristic.p, weeks_between=algo_config["r"]
    )

    return [metaheuristic_name, profit, duration]


def main_metaheuristics_controller(
    start_sol: np.ndarray,
    metaheuristics_to_use: list[str],
    algo_config: dict[str, Union[int, float, np.ndarray]],
    time_out: int,
):
    df_res = pd.DataFrame(columns=["Metaheuristic", "profit", "duration"])
    profit = compute_profit(
        sol=start_sol, profit=np.array(algo_config["p"]).reshape((3, algo_config['n'], algo_config['n'])), weeks_between=algo_config["r"]
    )
    df_res.loc[len(df_res)] = ['Start sol', profit, 0]

    for metaheuristic_name in metaheuristics_to_use:
        print(f"Executing {metaheuristic_name}")
        parameters = get_yaml_config(metaheuristic_name=metaheuristic_name)
        results = apply_metaheuristic(
            metaheuristic_name=metaheuristic_name,
            start_sol=start_sol,
            algo_config=algo_config,
            time_out=time_out,
            parameters=parameters,
        )
        df_res.loc[len(df_res)] = results
        print()

    print(df_res)
