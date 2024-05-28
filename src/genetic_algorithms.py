from datetime import timedelta
import time

# Third party library
import numpy as np

# Project specific library
from src.helper import print_solution

class GeneticAlgorithm():
    def __init__(self, algo_config) -> None:
        self.n = algo_config['n'] # int
        self.t = algo_config['t'] # float
        self.s = algo_config['s'] # int
        self.r = algo_config['r'] # int
        self.p = algo_config['p'] # np.array

    def run(self) -> np.array:
        """
        Execute algorithm

        Args:
           None

        Returns:
            None
        """
        pass

    def check_solution():
        """
        not feasible returns none
        is feasible returns np.array
        """
        pass

    def execute_cmd(self):
        # https://stackoverflow.com/a/61713634 28.05.2024
        start = time.perf_counter()
        solution_set = self.run()
        runtime = timedelta(seconds=time.perf_counter() - start)

        print_solution(runtime, solution_set)





