from datetime import timedelta
import time

# Third party library
import numpy as np
from copy import deepcopy
from collections import deque

# Project specific library
# from src.helper import print_solution
# from src.ValidationGregor import SolutionValidation, StepValidation, permutations
from helper import print_solution
from ValidationGregor import SolutionValidation, StepValidation, permutations

class GreedyHeuristic(SolutionValidation, StepValidation):
    def __init__(self, algo_config) -> None:
        self.n = algo_config['n'] # int
        self.t = algo_config['t'] # float
        self.r = algo_config['r'] # ints
        if self.input_p_is_correct(algo_config['p']):
            self.p = np.array(algo_config['p']).reshape(3,self.n,self.n) # np.array

        self.w = 2*(self.n - 1)
        # self.matrix = algo_config['p'].reshape(3,self.n,self.n)
        self.mpw = self.n * self.n
        # https://stackoverflow.com/a/1704853 
        self.maxMatchesDay = np.floor(self.n*self.t).astype(np.int16)
        self.allTeams = set(range(1,self.n+1))

        self.solution = np.zeros( (3, 2*(self.n - 1), np.floor(self.n*self.t).astype(np.int16), 2), dtype=np.int8) # day - week - matchOfDay - teamsMatch
        self.allMatches = set(map(lambda match: (int(match[0]), int(match[1])), permutations(self.allTeams, 2)))
        
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
        # number_matches = (self.n*self.n) - self.n
        self.assign_monday_games()


        helper = 0
        while helper < 3: #len(self.allMatches) != 0: #
            p_shape = self.get_current_profits()
            incomplete_weeks = self.get_incomplete_weeks()

            if len(incomplete_weeks) != 0:
                for week in list(incomplete_weeks)[:1]:
                    teams_left = self.get_missing_teams(week=week)
                    print(teams_left)
                    if self.allMatches.issubset(teams_left):
                        continue
                    self.fill_weeks_greedy(p_shape, week, teams_left)
            
            print_solution(0, self.solution)
            helper += 1


    def fill_weeks_greedy(self, p_shape, week, teams_left):
        allSlots = self.solution.transpose(1,0,2,3)[week]
        freeTeamSlots = np.argwhere(allSlots == 0)
        freeSlots = np.unique(freeTeamSlots[:,[0,1]], axis=0)

        # remove slots on monday that should not be taken
        forbiddenSlots = [[0,i] for i in range(1, self.maxMatchesDay)]
        freeSlots = np.array([row for row in freeSlots if list(row) not in forbiddenSlots])

        profits = deque()
        slots = deque()
        for idx, pm in enumerate(permutations(teams_left, 2)):
            poss_days = np.unique(freeSlots[:,0])
            pm_ii = np.array(pm)-1
            for pd in poss_days:
                profits.append(p_shape[pd,*pm_ii])
                slots.append([pd,*pm_ii])

        r_weeks = self.solution.transpose(1,0,2,3)[week+self.r : week+self.r+1]
        iRematches = r_weeks[r_weeks != [0, 0]].reshape(-1, 2)[:, [1,0]] - 1
        
        matchesPlayed = self.solution.transpose(1,0,2,3)[:]
        iMatchesPlayed = matchesPlayed[matchesPlayed != [0, 0]].reshape(-1, 2) - 1
        
        iRemoveMatches = np.append(iMatchesPlayed, iRematches, axis=0)
        
        slots = np.array([slot for slot in slots if np.all(np.any(slot[1:3] != iRemoveMatches, axis=1))])
        profits = np.array([profit for profit in profits if profit != -1])

        arg_profit = np.array(profits).argsort()[::-1]

        while self.check_imcomplete_week(week=week):
            iMaxProfit = arg_profit[0]
            sMaxProfit = slots[iMaxProfit]
            tMaxProfit = sMaxProfit[1:3] 
            dayWeek = self.solution.transpose(1,0,2,3)[week, sMaxProfit[0]]
            freeSlots = np.where(np.all(dayWeek == [0,0], axis=1))[0]
            teams = np.array(tMaxProfit) + 1

            self.solution.transpose(1,0,2,3)[week, sMaxProfit[0], freeSlots[0]] = teams 
            self.allMatches.remove(tuple(teams))
            
            matches = np.array(slots)[:, [1,2]]
            test1 = np.any(matches == tMaxProfit, axis=1)
            test2 = np.any(matches == tMaxProfit[::-1], axis=1)
            iCoveredTeams = np.unique(np.where(test1 | test2))

            arg_profit = np.setdiff1d(arg_profit, iCoveredTeams, assume_unique=True)

    def check_imcomplete_week(self, week):
        week = self.solution.transpose(1,0,2,3)[week]
        idx_teams = np.where(week.flatten() != 0)
        teams = set(week.flatten()[idx_teams])
        return not self.allTeams.issubset(teams)

    def get_incomplete_weeks(self):
        incomplete_weeks = deque()
        for idx_week, week in enumerate(self.solution.transpose(1,0,2,3)):
            idx_teams = np.where(week.flatten() != 0)
            teams = set(week.flatten()[idx_teams])
            if not self.allTeams.issubset(teams):
                incomplete_weeks.append(idx_week)
        return incomplete_weeks
    
    def get_missing_teams(self, week):
        allSlots = self.solution.transpose(1,0,2,3)[week]
        coveredTeams = set(allSlots.flatten()) - {0}        
        return self.allTeams.difference(coveredTeams)

    def get_current_profits(self):
        p_shape = deepcopy(self.p)
        non_zero_indices = np.argwhere(self.solution[:, :, :, 0] != 0)
        matches = [
            (self.solution[day, week, match, 0], self.solution[day, week, match, 1])
            for day, week, match in non_zero_indices
        ]
        for wd in range(3):
            for t1, t2 in matches:
                p_shape[wd, t1-1, t2-1] = -1

        return p_shape

    def assign_monday_games(self):
        p_shape = deepcopy(self.p)
        weekDay = 0  # Montag

        idxMaxMatches = p_shape[0].flatten().argsort()[::-1]
        monday_games = np.array(np.unravel_index(idxMaxMatches, (6,6)))
        monday_games = monday_games.reshape(2,-1).transpose(1,0) + 1

        for t1,t2 in monday_games:
            monFirstSlots = self.solution[weekDay].transpose(1,0,2)[0]
            if np.any(monFirstSlots == [t1,t2]) or np.any(monFirstSlots == [t2,t1]):
                continue

            freeSlots = np.all(monFirstSlots == [0,0], axis=1)
            firstFreeSlot = np.where(freeSlots)[0][0]
            self.solution[weekDay, firstFreeSlot, 0] = [t1,t2]        


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