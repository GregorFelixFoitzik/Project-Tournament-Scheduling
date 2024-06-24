import numpy as np
from collections import deque
from copy import deepcopy

class ConstructionHelper:
    ### TO

    ### SETTER
    def set_to_solution(self, week: int, day: int, slot: int, match: np.array):
        self.solution.transpose(1,0,2,3)[week, day, slot] = match 
        self.allMatches.remove(tuple(match))


    ### GETTER
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

    def get_total_incomplete_weeks(self):
        total_incomplete_weeks = deque()
        for idx_week, week in enumerate(self.solution.transpose(1,0,2,3)):
            idx_teams = np.where(week.flatten() != 0)
            teams = set(week.flatten()[idx_teams])
            if len(teams) == 0:
                total_incomplete_weeks.append(idx_week)
        return total_incomplete_weeks

    def get_partially_incomplete_weeks(self):
        partially_incomplete_weeks = deque()
        for idx_week, week in enumerate(self.solution.transpose(1,0,2,3)):
            idx_teams = np.where(week.flatten() != 0)
            teams = set(week.flatten()[idx_teams])
            if not self.__allTeams__.issubset(teams) and len(teams) != 0:
                partially_incomplete_weeks.append(idx_week)
        return partially_incomplete_weeks
    
    def get_missing_teams(self, week):
        allSlots = self.solution.transpose(1,0,2,3)[week]
        coveredTeams = set(allSlots.flatten()) - {0}        
        return self.__allTeams__.difference(coveredTeams)
    
    def get_rematches_in_r(self, week):
        r_start = np.maximum(0,week-self.__r__)
        r_end = np.minimum(week+self.__r__, self.__w__) + 1
        r_weeks = self.solution.transpose(1,0,2,3)[r_start : r_end]
        iRematches = r_weeks[r_weeks != [0, 0]].reshape(-1, 2)[:, [1,0]] - 1
        return np.array(iRematches, np.int16)

    def get_matches_played(self):
        matchesPlayed = self.solution.transpose(1,0,2,3)[:]
        iMatchesPlayed = matchesPlayed[matchesPlayed != [0, 0]].reshape(-1, 2) - 1
        return np.array(iMatchesPlayed, np.int16)
    
    #https://stackoverflow.com/a/64416484
    def get_allowed_matches(self, A, B):
        nrows, ncols = A.shape
        dtype={'names':['f{}'.format(i) for i in range(ncols)], 'formats':ncols * [A.dtype]}
        C = np.setdiff1d(A.copy().view(dtype), B.copy().view(dtype))
        return np.array( [list(m) for m in C] )
    
    def get_allowed_week_combs(self, comb):
        all_numbers = [num for subtuple in comb for num in subtuple]
        return len(set(all_numbers)) == len(all_numbers)
    
    ### CHECKER
    def check_imcomplete_week(self, week):
        week = self.solution.transpose(1,0,2,3)[week]
        idx_teams = np.where(week.flatten() != 0)
        teams = set(week.flatten()[idx_teams])
        return not self.__allTeams__.issubset(teams)