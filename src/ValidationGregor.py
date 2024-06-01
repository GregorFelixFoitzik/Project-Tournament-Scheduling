from itertools import permutations, product
from collections import Counter
import numpy as np

class Validation():
    def validate(self, sol, n, r, p, t, s):
        w = 2*(n-1)
        # self.team_plays_twice_home_aways(n, sol)
        # self.every_team_plays_every_week(n, sol)
        # self.one_match_on_monday(n, sol)
        # assert self.t_percent_on_friday_saturday(), "There are more than t-% games on a friday or saturday"
        # self.at_least_one_monday_match(n, sol)
        self.at_least_r_weeks_between(r, sol) #"There are less than r weeks between the first and second round"
        
    def team_plays_twice_home_aways(self, n: int, solution: np.array) -> None:
        all_teams = set(range(1, n + 1))             
        permuts = set(map(lambda match: f"{int(match[0])}vs{int(match[1])}", permutations(all_teams, 2))) ### generates all match that need to be played
        sol_set = [match for match in solution.flatten() if match != 'nan'] ### makes it easy to compare
        assert permuts == set(sol_set), "A team does not play once at home and once away" 

        for match, count in Counter(sol_set).items():
            assert count == 1, f"The match {match} appears more than once."

    def every_team_plays_every_week(self, n, solution) -> None:
        all_teams = set(range(1, n + 1))
        sol_teams_week = [
            {int(team) for match in week if match[0] != 'nan' for team in match[0].split("vs")}
            for week in solution
        ]
        
        for idx, teams_week in enumerate(sol_teams_week):
            assert all_teams.issubset(teams_week), f"At least one team does not play in week {idx + 1}"
        
    def input_p_is_correct(self, p) -> None:
        matrix = np.array(p).reshape(3,6,6)
        for sm in matrix:
            for i in range(sm.shape[0]):
                for j in range(sm.shape[1]):
                    assert not ((i != j) and (sm[i,j] < 0)), "The profit of a match is negative"
                    assert not ((i == j) and (sm[i,j] != -1)), "Matches against yourself must have a profit of -1"

    def only_one_match_on_monday(self, n, solution):
        ### This will be implemented within the calculation of the solution
        ### since we only take the first column of the solution matrix
        ### as mondays 
        pass

    def t_percent_on_friday_saturday(self, t, sol):
        ### TODO: Jonas pls help
        pass

    def at_least_one_monday_match(self, n, solution):
        all_teams = set(range(1, n + 1))
        teams_on_mondays = set([
            int(team) for monday in solution[:, 0]
            for team in monday[0].split("vs") if team != 'nan'
        ])
        assert teams_on_mondays == all_teams, "At least one team never plays on any monday"

    def at_least_r_weeks_between(self, r, solution):
        for idx_w, week in enumerate(solution):
            for match in week:
                if match[0] != 'nan':
                    second_round = match[0][::-1].replace("sv", "vs")
                    assert not second_round in solution[idx_w:idx_w+r+1], f"Less than r={r} weeks between first and second round of {match[0]} resp. {second_round}"


if __name__ == "__main__":
    val = Validation()

    num_teams = 6
    weeks_between = 1
    profits = [
        -1.0,
        15.0,
        20.0,
        16.0,
        28.0,
        9.0,
        13.0,
        -1.0,
        8.0,
        12.0,
        10.0,
        12.0,
        19.0,
        10.0,
        -1.0,
        14.0,
        15.0,
        13.0,
        18.0,
        14.0,
        12.0,
        -1.0,
        13.0,
        16.0,
        25.0,
        15.0,
        13.0,
        17.0,
        -1.0,
        11.0,
        10.0,
        14.0,
        23.0,
        21.0,
        10.0,
        -1.0,
        -1.0,
        8.0,
        10.0,
        9.0,
        5.0,
        9.0,
        6.0,
        -1.0,
        5.0,
        11.0,
        7.0,
        4.0,
        8.0,
        4.0,
        -1.0,
        12.0,
        8.0,
        6.0,
        10.0,
        7.0,
        4.0,
        -1.0,
        4.0,
        2.0,
        10.0,
        7.0,
        5.0,
        3.0,
        -1.0,
        10.0,
        4.0,
        8.0,
        2.0,
        7.0,
        1.0,
        -1.0,
        -1.0,
        2.0,
        3.0,
        5.0,
        3.0,
        5.0,
        1.0,
        -1.0,
        3.0,
        2.0,
        9.0,
        10.0,
        7.0,
        8.0,
        -1.0,
        4.0,
        5.0,
        2.0,
        6.0,
        7.0,
        5.0,
        -1.0,
        2.0,
        1.0,
        6.0,
        8.0,
        5.0,
        4.0,
        -1.0,
        7.0,
        8.0,
        3.0,
        2.0,
        7.0,
        4.0,
        -1.0,
    ]

    t = 2 / 3
    s = 2

    sol = np.array([
        [["5vs1"], ["3vs6"], ["2vs4"], [np.nan], [np.nan]],
        [["6vs2"], ["1vs4"], ["3vs5"], [np.nan], [np.nan]],
        [["5vs4"], ["1vs6"], [np.nan], ["3vs2"], [np.nan]],
        [["6vs4"], ["1vs3"], [np.nan], ["5vs2"], [np.nan]],
        [["1vs2"], ["3vs4"], ["5vs6"], [np.nan], [np.nan]],
        [["6vs3"], ["4vs1"], [np.nan], ["2vs5"], [np.nan]],
        [["3vs1"], ["4vs2"], [np.nan], ["6vs5"], [np.nan]],
        [["1vs5"], [np.nan], [np.nan], ["4vs3"], ["2vs6"]],
        [["4vs5"], ["2vs3"], [np.nan], ["6vs1"], [np.nan]],
        [["4vs6"], ["5vs3"], ["2vs1"], [np.nan], [np.nan]],
    ])

    val.input_p_is_correct(profits)
    val.validate(
        sol=sol,
        n=num_teams,
        r=weeks_between,
        p=profits,
        t=t,
        s=s,
    )
