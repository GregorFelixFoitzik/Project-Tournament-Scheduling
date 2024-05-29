from datetime import timedelta
import time

# Third party library
import numpy as np
import copy

# Project specific library
from src.helper import print_solution
from src.ValidationGregor import Validation

class GreedyHeuristic(Validation):
    def __init__(self, algo_config) -> None:
        self.n = algo_config['n'] # int
        self.t = algo_config['t'] # float
        self.s = algo_config['s'] # int
        self.r = algo_config['r'] # int
        self.p = algo_config['p'] # np.array
        assert(Validation.input_p_is_correct(self.p))

        self.w = 2*(self.n - 1)
        # self.matrix = algo_config['p'].reshape(3,self.n,self.n)
        self.gpw = self.n * self.n
        self.solution = np.array([])

    def run(self) -> np.array:
        """
        Execute algorithm

        Args:
           None

        Returns:
            None
        """
        mat = copy.deepcopy(self.p)
        vv_max = np.max(mat)
        ii_max = np.random.choice(np.where(mat == vv_max)[0])
        print(ii_max)
        # week  : 
        # day   : ii_max//self.gpw      ### weekday 0:M 1:F 2:S
        # ti    : ii_max//self.n +1     ### team i
        # tj    : ii_max%self.n +1      ### team j
        # pkij  : self.p[ii_max]
        print(
            f"""
day: {ii_max//self.gpw}
ti: {ii_max//self.n +1}
tj: {ii_max%self.n +1}
pkij: {self.p[ii_max]}
"""
        )


    def check_solution(self):
        """
        not feasible returns none
        is feasible returns np.array
        """
        self.team_plays_twice_home_aways(self.n, self.solution)
        self.every_team_place_every_week(self.n, self.w, self.solution)

    def execute_cmd(self):
        # https://stackoverflow.com/a/61713634 28.05.2024
        start = time.perf_counter()
        solution_set = self.run()
        runtime = timedelta(seconds=time.perf_counter() - start)

        print_solution(runtime, solution_set)