"""File contains helper functions which might be useful"""
# Standard library
from math import inf
import re
import ast

# Third party library
import numpy as np
from typing import Union
from itertools import permutations
from tabulate import tabulate


def read_in_file(path_to_file: str) -> dict[str, Union[float, int, list[int]]]:
    """Read the .in file which contains the configuration.

    After extracting the data, the strings are converted into int, floats and ints.

    Args:
        path_to_file (str): Where is the .in file stored?

    Returns:
        dict[str, Union[float, int, list[int]]]: .in file as dictionary.
    """
    with open(file=path_to_file, mode='r') as file:
        lines = file.readlines()

    algo_config = {}
    # String operations on each line
    for line in lines:
        algo_config[line.split(sep=':')[0]] = line.split(sep=':')[1]#.replace(r'\n', '').strip()

    # Converting the values of the .in file into int, float and list
    for key, value in algo_config.items():
        if re.search(pattern=r'\d*\/\d*', string=value):
            algo_config[key] = int(value.split('/')[0]) / int(value.split('/')[1])
        elif re.search(pattern=r'\[.*\]', string=value):
            # https://stackoverflow.com/a/1894296, accessed 28.05.2024
            algo_config[key] = np.array(ast.literal_eval(node_or_string=value))
        else:
            algo_config[key] = int(value)

    return algo_config



def print_solution(runtime: float, solution: np.array = None) -> None:
    """
    Prints the solution as wanted.

    Args:
        runtime (float): CPU-Runtime (in seconds)
        solution (list): contains the current best solution 
            index of list: 
            #     idx // (self.n/2) => week
            #     idx %  3 => M/F/S (0/1/2)
            value (list): [home, away, profit]
    """
    if solution is not None:
        formatted_blocks = []
        for block in solution:
            formatted_block = [[' - '.join(map(str, pair)).replace('0', '') for pair in row] for row in block]
            formatted_blocks.append(formatted_block)

        # Convert the formatted blocks into numpy arrays
        formatted_blocks = [np.array(block) for block in formatted_blocks]

        # Concatenate the blocks horizontally
        concatenated_array = np.hstack(formatted_blocks)

        # Print the concatenated array using tabulate
        headers=['Mon','Mon','Mon', 'Fri','Fri','Fri', 'Sat','Sat','Sat']
        print(tabulate(concatenated_array, tablefmt="grid", headers=headers))  

    else:
        print("The solution is None and cannot be processed.")

def compute_profit(sol: np.ndarray, profit: np.ndarray) -> float:
    return np.sum(get_profits_per_week(sol, profit))


def get_profits_per_week(sol: np.ndarray, profit: np.ndarray):
    week_profits = []
    for week in sol:
        sum_week = 0
        for i, matches in enumerate(week):
            if i == 0 and np.unique(matches, axis=0).shape[0] == 2:
                match = matches[np.logical_not(np.isnan(matches))]
                sum_week += profit[0][int(match[0])-1][int(match[1])-1]
                continue
            if i == 0 and np.unique(matches, axis=0).shape[1].shape[1] > 2:
                matches = matches[np.logical_not(np.isnan(matches))]
                min_profit = float(inf)

                for match in matches:
                     if profit[0][int(match[0])-1][int(match[1])-1] < min_profit:
                          min_profit = profit[0][int(match[0])-1][int(match[1])-1]
                sum_week += min_profit
                continue
            
            for match in matches:
                if False in np.isnan(match):
                    sum_week+= profit[i][int(match[0])-1][int(match[1])-1]
        week_profits.append(sum_week)

    return week_profits
    
def get_profit_games_earlier(profit: float, q: np.ndarray)->np.ndarray:
    return profit / (1+q^2)