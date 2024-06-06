from datetime import timedelta
import time

# Third party library
import numpy as np
from copy import deepcopy
# from collections import deque

# Project specific library
from src.helper import print_solution
from src.ValidationGregor import SolutionValidation, StepValidation, permutations

class GreedyHeuristic(SolutionValidation, StepValidation):
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

        self.solution = np.zeros( (3, 2*(self.n - 1), self.n//2, 2), dtype=np.int8) # day - week - matchOfDay - teamsMatch
        
    def run(self) -> np.array:
        """
        Execute algorithm

        Args:
           None

        Returns:
            None
        """

        self.NaivGreedy()

    def NaivGreedy(self) -> str:
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
        allMatches = set(map(lambda match: (int(match[0]), int(match[1])), permutations(set(range(1, self.n + 1)), 2)))
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

        profits = deepcopy(self.p)

        while len(allMatches) != 0:

            profitMaxMatch = np.max(profits)
            idxMaxMatches = np.where( profitMaxMatch == profits )[0]
            #     HelperProfitStart[0]:
            #     HelperProfitEnd[0]            
            # ])[0]

            for idxMaxMatch in idxMaxMatches:
                weekday, t1, t2 = np.unravel_index(idxMaxMatch, (3,self.n,self.n))
                t1, t2 = t1+1, t2+1

                if weekday == 0:                    
                    possWeeks = [ii for ii, tt in enumerate(self.solution[weekday][:, 0]) if all(tt == [0,0])]
                    
                else:
                    possWeeks = [ii for ii, tt in enumerate(self.solution[weekday][:, :]) if all(tt == [0,0])]

            print(possWeeks)

            self.solution[weekday][possWeeks[0], 0] = [t1, t2]
            print(allMatches)
            allMatches.remove((t1,t2))
            profits[idxMaxMatch] = -1
            print(allMatches)

            print_solution(0, self.solution)

            # break
            

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