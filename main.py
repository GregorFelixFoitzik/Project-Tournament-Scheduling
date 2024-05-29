# Standard library
import argparse

# Project specific library
from src.helper import read_in_file
from src.genetic_algorithms import GeneticAlgorithm
from src.greedy_heuristic import GreedyHeuristic

# Copied from https://docs.python.org/3/library/argparse.html, accessed 28.05.2024
parser = argparse.ArgumentParser(description="Input for algorithms")
# ChatGPT fixed the issues that default values were not used. => python main.py works now
parser.add_argument("-t", "--timeout", type=int, default=30, help="Timeout duration in seconds")
parser.add_argument("-p", "--path_to_instance", type=str, default="data/example.in", help="Path to the input file")

if __name__ == '__main__':
    # Extract command line arguments
    args = parser.parse_args()
    path_to_file = args.path_to_instance
    timeout = args.timeout

    algo_config = read_in_file(path_to_file=path_to_file)
       
    agent = GreedyHeuristic(algo_config)
    agent.execute_cmd()