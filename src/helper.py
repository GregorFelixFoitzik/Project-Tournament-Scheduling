"""File contains helper functions which might be useful"""
# Standard library
import re
import ast

# Third party library
import numpy as np
from typing import Union
from itertools import permutations

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



def print_solution(self, runtime: float, solution: np.array = None) -> None:
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
    if solution:
        objective_value = np.sum(solution, axis=0)[-1]
        print("### RESULT: Feasible")
        print(f"### OBJECTIVE: {objective_value}")

        for idx, match in enumerate(solution):
            days = {0: 'M', 1: 'F', 2: 'S'}
            print(f"### match <{idx//(self.n/2)+1}>-{days[idx%3]}: <{match[0]}> <{match[1]}>")

        print(f"### CPU-TIME: {runtime}")

    else:
        print("### RESULT: TIMEOUT")