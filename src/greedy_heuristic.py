from datetime import timedelta
import time

# Third party library
import numpy as np
from copy import deepcopy
from collections import deque

# Project specific library
# from src.helper import print_solution
# from src.ValidationGregor import SolutionValidation, permutations
# from src.ConstructionHelper import ConstructionHelper
from helper import print_solution
from ValidationGregor import SolutionValidation, permutations
from ConstructionHelper import ConstructionHelper

class GreedyHeuristic(SolutionValidation, ConstructionHelper):
    def __init__(self, algo_config) -> None:
        self.__n__ = algo_config['n'] # int
        self.__t__ = algo_config['t'] # float
        self.__r__ = algo_config['r'] # ints
        self.__w__ = 2*(self.__n__ - 1)
        self.__mpw__ = self.__n__ * self.__n__
        self.__maxMatchesDay__ = np.floor(self.__n__*self.__t__).astype(np.int16)
        self.__allTeams__ = set(range(1,self.__n__+1))

        if self.input_p_is_correct(algo_config['p']):
            self.p = np.array(algo_config['p']).reshape(3,self.__n__,self.__n__) # np.array
        self.solution = np.zeros( (3, 2*(self.__n__ - 1), np.floor(self.__n__*self.__t__).astype(np.int16), 2), dtype=np.int8) # day - week - matchOfDay - teamsMatch
        self.allMatches = set(map(lambda match: (int(match[0]), int(match[1])), permutations(self.__allTeams__, 2)))
        
    def run(self) -> np.array:
        """ EXECUTES ALGORITHM
        Selects the match with the highest profit over every day.

        Args:
            None                         
        
        Returns:
            None
        """
        # number_matches = (self.__n__*self.__n__) - self.__n__
        self.assign_monday_games()
        p_shape = self.get_current_profits()
        # total_incomplete_weeks = self.get_total_incomplete_weeks()
        # for week in total_incomplete_weeks:
        #     idxMaxMatches = p_shape.flatten().argsort()[::-1]
        #     idxNegProfit = np.where(p_shape.flatten() == -1)[0]
        #     idxMaxMatches = np.setdiff1d(idxMaxMatches, idxNegProfit, assume_unique=True)
            
        #     iRematches = self.get_rematches_in_r(week) 
        #     iMatchesPlayed = self.get_matches_played()
        #     matchesNotAllowed = np.append(iMatchesPlayed, iRematches, axis=0)
        #     idxFlatNotAllowedMon = matchesNotAllowed[:,0] * 6 + matchesNotAllowed[:,1]
        #     idxFlatNotAllowed = np.append(idxFlatNotAllowedMon, idxFlatNotAllowedMon+(self.__n__*self.__n__))
        #     idxFlatNotAllowed = np.append(idxFlatNotAllowed, idxFlatNotAllowedMon+(self.__n__*self.__n__)*2)

        #     idxMaxMatches = np.setdiff1d(idxMaxMatches, idxFlatNotAllowed, assume_unique=True)
            
        #     day, t1, t2 = np.array(np.unravel_index(idxMaxMatches[0], (3,6,6)))
        #     self.set_to_solution(week, day, 0, np.array([t1,t2])+1)
        
        helper = 0
        while len(self.allMatches) != 0: #helper < 1: #
            iMatchesPlayed = self.get_matches_played()        
            iRematches = self.get_rematches_in_r(0)            
            p_shape = self.get_current_profits()
            
            partially_incomplete_weeks = self.get_partially_incomplete_weeks()
            # print('partially incomplete', partially_incomplete_weeks)
            if len(partially_incomplete_weeks) != 0:
                for week in list(partially_incomplete_weeks)[:1]:
                    teams_left = self.get_missing_teams(week=week)
                    if self.allMatches.issubset(teams_left):
                        continue
                    self.fill_weeks_greedy(p_shape, week, teams_left)
            
            print_solution(0, self.solution)
        helper += 1

    def assign_monday_games(self):
        p_shape = deepcopy(self.p)
        weekDay = 0  # Montag
        idxMaxMatches = p_shape[weekDay].flatten().argsort()[::-1]
        monday_matches = np.array(np.unravel_index(idxMaxMatches, (6,6)))
        monday_matches = monday_matches.reshape(2,-1).transpose(1,0) + 1

        for t1,t2 in monday_matches:
            monFirstSlots = self.solution[weekDay].transpose(1,0,2)[0]
            if np.any(monFirstSlots == [t1,t2]) or np.any(monFirstSlots == [t2,t1]):
                continue

            freeSlots = np.all(monFirstSlots == 0, axis=1)
            firstFreeSlot = np.where(freeSlots)[0][0]
            # print(t1,t2)
            # self.set_to_solution(weekDay, 0, firstFreeSlot, [t1,t2])
            self.solution[weekDay, firstFreeSlot, 0] = [t1,t2]
            self.allMatches.remove((t1,t2))

    def fill_weeks_greedy(self, p_shape, week, teams_left):
        allSlots = self.solution.transpose(1,0,2,3)[week]
        freeTeamSlots = np.argwhere(allSlots == 0)
        freeSlots = np.unique(freeTeamSlots[:,[0,1]], axis=0)
        forbiddenSlots = [[0,i] for i in range(1, self.__maxMatchesDay__)]
        freeSlots = np.array([row for row in freeSlots if list(row) not in forbiddenSlots])
        freeDays = np.unique(freeSlots[:,0])
        
        missingTeams = self.get_missing_teams(week)
        iRematches = self.get_rematches_in_r(week) 
        iMatchesPlayed = self.get_matches_played()
        possibleMatches = np.array(list(permutations(missingTeams, 2)), np.int16)
        allowedMatches = self.get_allowed_matches(possibleMatches, iRematches+1)
        allowedMatches = self.get_allowed_matches(allowedMatches, iMatchesPlayed+1)
        print(allowedMatches)
        print('missing teams before while', missingTeams)

        matchInfo = np.zeros(
            (   allowedMatches.shape[0]*3,
                allowedMatches.shape[1]+2   )
        ,dtype=np.int16)

        for dd in freeDays:
            matchInfo[allowedMatches.shape[0]*dd : allowedMatches.shape[0]*(dd+1), 0] = dd
            matchInfo[allowedMatches.shape[0]*dd : allowedMatches.shape[0]*(dd+1),[1,2]] = allowedMatches-1
            print('MI1', matchInfo)

        for i, m in enumerate(matchInfo):
            matchInfo[i, 3] = p_shape[m[0], m[1], m[2]]

        arg_profit = np.argsort(matchInfo[:, 3])[::-1]
        print('info before while', matchInfo)

        while self.check_imcomplete_week(week=week):
            print(week, self.check_imcomplete_week(week))
            # print_solution(0, self.solution)
            iMaxProfit = arg_profit[0]
            print('imaxprofit', iMaxProfit)
            dMaxProfit = matchInfo[iMaxProfit, 0]
            print('dmaxprofit', dMaxProfit)
            tMaxProfit = matchInfo[iMaxProfit, [1,2]]
            print('tmaxprofit', tMaxProfit)
            dayWeek = self.solution.transpose(1,0,2,3)[week, dMaxProfit]
            freeSlots = np.where(np.all(dayWeek == [0,0], axis=1))[0]
            teams = tMaxProfit + 1
            print(teams)

            matches = matchInfo[:, [1,2]]
            test1 = np.any(matches == tMaxProfit, axis=1)
            test2 = np.any(matches == tMaxProfit[::-1], axis=1)
            iCoveredTeams = np.unique(np.where(test1 | test2))
            
            arg_profit = np.setdiff1d(arg_profit, iCoveredTeams, assume_unique=True)
            print('profit updated', matchInfo[arg_profit])
            
            self.set_to_solution(week, dMaxProfit, freeSlots[0], teams)
            print_solution(0, self.solution)

            print(week, 'missingteams', self.get_missing_teams(week=week))


    def check_solution(self):
        """
        not feasible returns none
        is feasible returns np.array
        """
        if all(
            self.__t__eam_plays_twice_home_aways(self.__n__, self.solution),
            self.every_team_place_every_week(self.__n__, self.__w__, self.solution)
        ):
            return True

    def execute_cmd(self):
        # https://stackoverflow.com/a/61713634 28.05.2024
        start = time.perf_counter()
        solution_set = self.run()
        runtime = timedelta(seconds=time.perf_counter() - start)

        print_solution(runtime, solution_set)






if __name__ == "__main__":
    # Standard library
    import argparse
    import numpy as np

    # Project specific library
    # from src.helper import read_in_file
    # from src.genetic_algorithms import GeneticAlgorithm
    # from src.greedy_heuristic import GreedyHeuristic
    from helper import read_in_file
    # from genetic_algorithms import GeneticAlgorithm
    from greedy_heuristic import GreedyHeuristic

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
        # print(
        #     algo_config['p'].reshape(3,6,6)
        # )
        agent = GreedyHeuristic(algo_config)
        agent.execute_cmd()