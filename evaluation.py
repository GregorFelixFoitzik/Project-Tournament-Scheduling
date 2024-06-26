# Standard library
import os
import time

from typing import Union

# Third party libraries
import yaml
import numpy as np
import csv
from tqdm import tqdm


# Project specific library
from src.helper import (
    compute_profit,
    generate_solution_round_robin_tournament,
    read_in_file
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
    # "vns": VNS,
}


def get_yaml_config(mh_config_file: str) -> dict[str, Union[int, float]]:
    """Load the yaml-config file for a given metaheuristic.

    Args:
        metaheuristic_name (str): Name of the metaheuristic.

    Returns:
        dict[str, Union[int, float]]: Dictionary containing the configuration.
    """
    with open(file=f"configs/{mh_config_file}.yaml", mode="r") as file:
        config = yaml.safe_load(stream=file)

    return config["parameters"]

def apply_metaheuristic(
    metaheuristic_name: str,
    # start_sol: np.ndarray,
    algo_config: dict[str, Union[int, float, np.ndarray]],
    timeout: float,
    parameters: dict[str, Union[int, float]],
) -> list[Union[str, float]]:
    """Apply a metaheuristic to a given problem.

    Args:
        metaheuristic_name (str): Name of the metaheuristic.
        start_sol (np.ndarray): Start solution for a given problem.
        algo_config (dict[str, Union[int, float, np.ndarray]]): Configuration of the
            algorithm.
        timeout (float): Timeout for the Metaheuristic.
        parameters (dict[str, Union[int, float]]): Paramter-configuration for the
            Metaheuristic.

    Returns:
        list[Union[str, float]]: List containing the name, profit and duration.
    """
    start_sol, rc = generate_solution_round_robin_tournament(
        num_teams=int(algo_config["n"]),
        t=float(algo_config["t"]),
        random_team_order=False,
    )
    metaheuristic = METAHEURISTICS[metaheuristic_name](
        algo_config=algo_config,
        timeout=timeout,
        start_solution=start_sol,
        runtime_construction=rc
        **parameters,
    )
    new_sol = metaheuristic.run()
    duration = sum(os.times()[:2])

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


timeout = 30
instances = ['data/'+f for f in os.listdir('data/') if f.endswith('.in')]
config_files = [f.split('.')[0] for f in os.listdir('configs/') if f.endswith('.yaml')]

for mh in tqdm(METAHEURISTICS):
    meta_config_files = [f for f in config_files if mh == f.split('---')[0]]
    
    for instance in tqdm(instances):
        algo_config = read_in_file(path_to_file=instance)
 
        for config in meta_config_files:
            parameters = get_yaml_config(
                mh_config_file=config
            )
            results = apply_metaheuristic(
                metaheuristic_name=mh
                # ,start_sol=''
                ,algo_config=algo_config,
                timeout=timeout,
                parameters=parameters
            )

            # ### hier die csv
            # if not os.path.exists("artifacts/evaluations_first_run.csv"):
            #     os.mknod("artifacts/evaluations_first_run.csv")
            # row = [
            #     metaheuristic,
            #     algo_config,
            #     timeout,
            #     parameters,
            #     results[1],
            #     results[2],
            #     results[3],
            # ]
            # with open("artifacts/evaluations_first_run.csv", "a") as f:
            #     # [metaheuristic_name, profit, duration]
            #     """MetaName, AlgoConfig, Timeout, Parameters, StartSol_Profit, MH_Profti, MH_Time"""
            #     writer = csv.writer(f)
            #     writer.writerow(row)