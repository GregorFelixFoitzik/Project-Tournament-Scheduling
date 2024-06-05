from datetime import timedelta
import time

# Third party library
import numpy as np
import copy
# from collections import deque

# Project specific library
from src.helper import print_solution
from src.ValidationGregor import Validation

class GreedyHeuristic(Validation):
    def __init__(self, algo_config) -> None:
        self.n = algo_config['n'] # int
        self.t = algo_config['t'] # float
        self.r = algo_config['r'] # ints
        if self.input_p_is_correct(algo_config['p']):
            self.p = algo_config['p'] # np.array

        self.w = 2*(self.n - 1)
        # self.matrix = algo_config['p'].reshape(3,self.n,self.n)
        self.mpw = self.n * self.n
        # https://stackoverflow.com/a/1704853 
        self.solution = np.empty( (3, 2*(self.n - 1), self.n//2), dtype=str) # day - week - matchOfDay
        
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
        ii_max = np.where(mat == vv_max)[0][0] # np.random.choice(np.where(mat == vv_max)[0])
        self.NaivGreedyHeuristic(0, self.p)

    def NaivGreedyHeuristic(self, wd:int, profits:np.array) -> str:
        """
        Selects the match with the highest profit over every day.

        Args:
            wd:int
                - indicates the weekday {0: 'M', 1:'F', 2:'S'}
            profits:np.array
                - contains profits of all matches
                - profits of matches that are already "played" or are not possible due to conditions are equal to -1                         
        
        Returns:
            str : contains the match
        """
        weekdays = {0: 'M', 1:'F', 2:'S'}
        HelperProfitStart =  {
            0: 0
            ,1: self.n**2
            ,2: 2 * self.n**2
        }
        HelperProfitEnd = {
            0: self.n**2
            ,1: 2 * self.n**2
            ,2: 3 * self.n**2
        }
        idxMaxMatch = np.argmax(
            profits[
                HelperProfitStart[wd]:
                HelperProfitEnd[wd]            
            ]
        )
        print(profits)
        teams = np.unravel_index(idxMaxMatch, (3,self.n,self.n))
        S


    def check_solution(self):
        """
        not feasible returns none
        is feasible returns np.array
        """
        if all(
            self.team_plays_twice_home_aways(self.n, self.solution),
            self.every_team_place_every_week(self.n, self.w, self.solution)
        ):
            return True

    def execute_cmd(self):
        # https://stackoverflow.com/a/61713634 28.05.2024
        start = time.perf_counter()
        solution_set = self.run()
        runtime = timedelta(seconds=time.perf_counter() - start)

        print_solution(runtime, solution_set)