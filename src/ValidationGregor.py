from itertools import permutations, product
import numpy as np

class Validation:
    def team_plays_twice_home_aways(n, solution) -> bool:
        teams = np.linspace(1,n, n)                         
        permuts = set(permutations(teams, 2))               ## generates all match that need to be played
        solution = {tuple(match[:2]) for match in solution}  ## makes it easy to compare
        return permuts.issubset(solution)                   ## if all permutations are a subset of the solution, we cover all needed matchs

    def every_team_plays_every_week(n, w, solution) -> bool:
        teams = np.linspace(1,n,n)
        weeks = np.linspace(1,w,w)
        week_team_combs = set(product(weeks, teams))

        combs_solution = set()
        for idx, match in enumerate(solution):
            combs_solution.add( (idx//(n/2)+1, float(match[0])) )
            combs_solution.add( (idx//(n/2)+1, float(match[1])) )        
        return week_team_combs.issubset(combs_solution)

    def input_p_is_correct(p) -> bool:
        matrix = np.array(p).reshape(3,6,6)
        for sm in matrix:
            for i in range(sm.shape[0]):
                for j in range(sm.shape[1]):
                    if (i != j) and (sm[i,j] < 0):
                        return False
                    if (i == j) and (sm[i,j] != -1):
                        return False
        return True

    def one_match_on_monday():
        pass

    def t_percent_on_friday_saturday():
        pass

    def at_least_one_monday_match():
        pass

    def at_least_r_weeks_between():
        pass

solution = [
    [1,2, -1],
    [3,4, -1],
    [2,1, -1],
    [4,3, -1],
    
    [2,3, -1],
    [1,4, -1],
    [3,2, -1],
    [4,1, -1],
    
    [4,2, -1],
    [2,4, -1],
    [3,1, -1],
    [3,1, -1]
]
n = 4
w = 2*(n-1)

p = [-1.0, 15.0, 20.0, 16.0, 28.0, 9.0, 13.0, -1.0, 8.0, 12.0, 10.0, 12.0, 19.0, 10.0, -1.0, 14.0, 15.0, 13.0, 18.0, 14.0, 12.0, -1.0, 13.0, 16.0, 25.0, 15.0, 13.0, 17.0, -1.0, 11.0, 10.0, 14.0, 23.0, 21.0, 10.0, -1.0, -1.0, 8.0, 10.0, 9.0, 5.0, 9.0, 6.0, -1.0, 5.0, 11.0, 7.0, 4.0, 8.0, 4.0, -1.0, 12.0, 8.0, 6.0, 10.0, 7.0, 4.0, -1.0, 4.0, 2.0, 10.0, 7.0, 5.0, 3.0, -1.0, 10.0, 4.0, 8.0, 2.0, 7.0, 1.0, -1.0, -1.0, 2.0, 3.0, 5.0, 3.0, 5.0, 1.0, -1.0, 3.0, 2.0, 9.0, 10.0, 7.0, 8.0, -1.0, 4.0, 5.0, 2.0, 6.0, 7.0, 5.0, -1.0, 2.0, 1.0, 6.0, 8.0, 5.0, 4.0, -1.0, 7.0, 8.0, 3.0, 2.0, 7.0, 4.0, -1.0]

# def input_p_is_correct(p) -> bool:
#     matrix = np.array(p).reshape(3,6,6)

#     for sub_matrix in matrix:
#         if all(np.diag(sub_matrix) == -1):
#             return False
#     for i in range(matrix.shape[0]):
#         for j in range(matrix.shape[1]):
#             if (i != j) and (matrix[i,j] < 0):
#                 return False

#     return True
        
# input_p_is_correct(n, p)