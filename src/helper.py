"""File contains helper functions which might be useful"""
# Standard library
import re
import ast


from typing import Union

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
        algo_config[line.split(sep=':')[0]] = line.split(sep=':')[1].replace(r'\n', '').strip()

    # Converting the values of the .in file into int, float and list
    for key, value in algo_config.items():
        if re.search(pattern=r'\d*\/\d*', string=value):
            algo_config[key] = int(value.split('/')[0]) / int(value.split('/')[1])
        elif re.search(pattern=r'\[.*\]', string=value):
            # https://stackoverflow.com/a/1894296, accessed 28.05.2024
            algo_config[key] = ast.literal_eval(node_or_string=value)
        else:
            algo_config[key] = int(value)

    return algo_config
