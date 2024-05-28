# Standard library
import argparse

# Project specific library
from src.helper import read_in_file

# Copied from https://docs.python.org/3/library/argparse.html, accessed 28.05.2024
parser = argparse.ArgumentParser(description="Input for algorithms")
# TODO: @Jonas Ginster Remove default and replace --ARG with ARG
parser.add_argument("-t", "--timeout", default=30)
parser.add_argument("-p", "--path_to_instance", default="data/example.in")

if __name__ == '__main__':
    # Extract command line arguments
    args = parser.parse_args()
    path_to_file = args.path_to_instance
    timeout = args.timeout

    print(read_in_file(path_to_file=path_to_file))
