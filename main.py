# Standard library
import argparse

# Project specific library
from src.helper import read_in_file

# Copied from https://docs.python.org/3/library/argparse.html, accessed 28.05.2024
parser = argparse.ArgumentParser(description="Input for algorithms")
parser.add_argument("timeout")
parser.add_argument("path_to_instance")

def test2() -> None:
    pass 
def test1():
    print('Helloooooo, maybe this will result in a conflict')

if __name__ == '__main__':
    # Extract command line arguments
    args = parser.parse_args()
    path_to_file = args.path_to_instance
    timeout = args.timeout

    print(read_in_file(path_to_file=path_to_file))
