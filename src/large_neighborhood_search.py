# Standard library
import copy
import time
import random

from datetime import timedelta
import trace
from typing import Union

# Third party library
import numpy as np

from tabulate import tabulate


# Project specific library
from validation import each_team_on_monday, validate
from helper import compute_profit, get_profits_per_week, print_solution


class ALNS:
    def __init__(
        self,
        algo_config: dict[str, Union[int, float, np.ndarray]],
        time_out: int,
        start_solution: np.ndarray,
    ) -> None:
        self.n = algo_config["n"]  # int
        self.t = algo_config["t"]  # float
        self.s = algo_config["s"]  # int
        self.r = algo_config["r"]  # int
        # Has shape (3, n, n), so self.p[0] is the profit for monday
        self.p = np.array(algo_config["p"]).reshape((3, self.n, self.n))  # np.array

        self.time_out = time_out
        self.sol = start_solution
        self.best_solution = start_solution

        self.all_teams = range(1, self.n + 1)

    def run(self) -> np.ndarray:
        """
        Execute algorithm

        Args:
           None

        Returns:
            None
        """
        start_solution = self.sol.copy()
        best_solution = start_solution
        profit_best_solution = compute_profit(best_solution, self.p)

        t0 = time.time()
        while time.time() - t0 < self.time_out:
            sol_destroyed, matches, weeks_changed = self.destroy(best_solution.copy())
            new_sol = self.repair(sol_destroyed, matches, weeks_changed)
            profit_new_sol = compute_profit(new_sol, np.array(self.p))

            if profit_new_sol > profit_best_solution:
                best_solution = new_sol.copy()
                profit_best_solution = profit_new_sol

        print(f"Took: {time.time() - t0} {self.time_out}")

        self.best_solution = best_solution

        return best_solution

    def destroy(self, sol: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
        # Randomly choose a destroy parameter
        num_destroy_operators = 2
        destroy_operators = list(range(num_destroy_operators))
        weights = [100] * num_destroy_operators
        destroy_operator = random.choices(destroy_operators, weights=weights)[0]

        weeks_changed = []

        destroy_operator=1

        if destroy_operator == 0:
            # Destroy one week randomly
            week_to_destroy = np.random.choice(list(range(sol.shape[0])))
            matches = sol[week_to_destroy].copy()
            sol[week_to_destroy] = np.full(matches.shape, np.nan)
            weeks_changed = [week_to_destroy]
            matches = np.array([matches])
        elif destroy_operator == 1:
            # Destroy the 10$-worst solutions
            num_solutions_to_destroy = int(sol.shape[0] * 0.8)
            week_profits = get_profits_per_week(sol, self.p)

            worst_weeks = np.argsort(week_profits)[:num_solutions_to_destroy]
            matches = sol[worst_weeks].copy()
            
            sol[worst_weeks] = np.full(matches.shape, np.nan)
            weeks_changed = worst_weeks
        else:
            matches = np.array([])
            weeks_changed = np.array([])

        return sol, matches, weeks_changed

    def repair(self, sol: np.ndarray, matches: np.ndarray, weeks_changed: list[int]):
        # Randomly choose a repai parameter
        num_repair_operators = 2
        repair_operators = list(range(num_repair_operators))
        weights = [100] * num_repair_operators
        repair_operator = random.choices(repair_operators, weights=weights)[0]

        repair_operator = 1
        

        if repair_operator == 0:
            # Random fill
            for i, week_changed in enumerate(weeks_changed):
                matches = matches[i]
                new_order = np.full(sol.shape[1:], np.nan)

                teams_not_on_monday = np.setdiff1d(
                    self.all_teams,
                    np.sort(np.unique(sol[:, 0])[:-1].astype(int))[:-1],
                )

                matches_unique = np.unique(matches, axis=1)
                matches_unique = matches_unique[
                    np.logical_not(np.isnan(matches_unique))
                ]
                matches_unique = matches_unique.reshape(
                    int(matches_unique.shape[0] / 2), 2
                )
                if teams_not_on_monday.size != 0:
                    # Source: https://stackoverflow.com/a/38974252. accessed 8.6.2024
                    monday_idx = np.array(
                        np.where(np.isin(matches, teams_not_on_monday[0]))
                    ).reshape(1, -1)[0][:2]
                    monday_game = matches[monday_idx[0]][monday_idx[1]]
                else:
                    monday_game = random.choice(matches_unique)
                new_order[0][0] = monday_game

                matches_unique = matches_unique[
                    np.logical_not(np.isin(matches_unique, monday_game))
                ]
                matches_unique = matches_unique.reshape(
                    int(matches_unique.shape[0] / 2), 2
                )

                matches_unique = matches_unique[matches_unique != monday_game]

                possible_games = np.append(matches_unique, [np.nan, np.nan]).reshape(
                    int((matches_unique.shape[0] + 2) / 2), 2
                )
                friday_idx = range(possible_games.shape[0])
                friday_games = np.random.choice(
                    friday_idx, (new_order.shape[1],), replace=False
                )
                friday = possible_games[friday_games]

                saturday_idx = np.setdiff1d(friday_idx, friday_games)
                if saturday_idx.shape[0] < new_order.shape[1]:
                    saturday = possible_games[saturday_idx]
                    saturday = np.append(
                        saturday,
                        np.full(
                            (new_order.shape[1] - saturday_idx.shape[0], 2), np.nan
                        ),
                        axis=0,
                    )
                else:
                    saturday = np.sort(
                        np.random.choice(
                            np.setdiff1d(possible_games, friday),
                            (new_order.shape[1],),
                            replace=False,
                        )
                    )

                new_order[1] = friday
                new_order[2] = saturday

                sol[week_changed] = new_order
        elif repair_operator == 1:
            # Using worst weeks to generate a better solution
            matches_unique = np.unique(matches, axis=1)
            matches_unique = matches_unique[
                np.logical_not(np.isnan(matches_unique))
            ]
            matches_unique = matches_unique.reshape(
                int(matches_unique.shape[0] / 2), 2
            )

            # Get possible games for each of the empty weeks
            teams = np.unique(matches_unique)
            possible_weeks_per_team = []
            for team in teams:
                occurences = np.where(team == sol)[0]
                possible_weeks = []
                for destroyed_week in weeks_changed:
                    intersections = np.intersect1d(range(max(0, destroyed_week -self.r+1), min(destroyed_week +self.r, sol.shape[0])), occurences)
                    if intersections.size == 0:
                        possible_weeks.append(destroyed_week)
                possible_weeks_per_team.append(possible_weeks)
            
            teams_on_monday = np.unique(sol[:, 0])[:-1]
            teams_forced_on_monday = np.setdiff1d(teams, teams_on_monday)

            # 1. Matches forced on monday: Insert them first
            # 2. Find matches, that can only played in one week: Insert game, where profit is maximized
            # 3. Remaining games: Find maximum and fill up to t%

            # Add teams forced on monday

            games_forces_monday = []


        return sol

    def check_solution(self):
        """
        not feasible returns none
        is feasible returns np.array
        """
        validation = validate(self.sol, self.n, self.r, self.p, self.t, self.s)
        assert validation == True

    def execute_cmd(self):
        # https://stackoverflow.com/a/61713634 28.05.2024
        start = time.perf_counter()
        solution_set = self.run()
        runtime = timedelta(seconds=time.perf_counter() - start)

        print_solution(runtime, solution_set)


if __name__ == "__main__":
    num_teams = 6

    # sol = np.array(
    #     [
    #         [
    #             [[5, 1], [np.nan, np.nan]],
    #             [[3, 6], [2, 4]],
    #             [[np.nan, np.nan], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[6, 2], [np.nan, np.nan]],
    #             [[1, 4], [3, 5]],
    #             [[np.nan, np.nan], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[5, 4], [np.nan, np.nan]],
    #             [[1, 6], [np.nan, np.nan]],
    #             [[3, 2], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[6, 4], [np.nan, np.nan]],
    #             [[1, 3], [np.nan, np.nan]],
    #             [[5, 2], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[1, 2], [np.nan, np.nan]],
    #             [[3, 4], [5, 6]],
    #             [[np.nan, np.nan], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[6, 3], [np.nan, np.nan]],
    #             [[4, 1], [np.nan, np.nan]],
    #             [[2, 5], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[3, 1], [np.nan, np.nan]],
    #             [[4, 2], [np.nan, np.nan]],
    #             [[6, 5], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[1, 5], [np.nan, np.nan]],
    #             [[4, 3], [2, 6]],
    #             [[np.nan, np.nan], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[4, 5], [np.nan, np.nan]],
    #             [[2, 3], [np.nan, np.nan]],
    #             [[6, 1], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[4, 6], [np.nan, np.nan]],
    #             [[5, 3], [2, 1]],
    #             [[np.nan, np.nan], [np.nan, np.nan]],
    #         ],
    #     ]
    # )
    sol = np.array(
        [
            [
                [[5, 1], [np.nan, np.nan]],
                [[3, 2], [np.nan, np.nan]],
                [[6, 4], [np.nan, np.nan]],
            ],
            [
                [[6, 2], [np.nan, np.nan]],
                [[1, 4], [np.nan, np.nan]],
                [[3, 5], [np.nan, np.nan]],
            ],
            [
                [[1, 6], [np.nan, np.nan]],
                [[2, 4], [np.nan, np.nan]],
                [[5, 3], [np.nan, np.nan]],
            ],
            [
                [[1, 3], [np.nan, np.nan]],
                [[5, 2], [np.nan, np.nan]],
                [[4, 6], [np.nan, np.nan]],
            ],
            [
                [[1, 2], [np.nan, np.nan]],
                [[3, 4], [np.nan, np.nan]],
                [[5, 6], [np.nan, np.nan]],
            ],
            [
                [[4, 1], [np.nan, np.nan]],
                [[2, 5], [np.nan, np.nan]],
                [[6, 3], [np.nan, np.nan]],
            ],
            [
                [[1, 5], [np.nan, np.nan]],
                [[4, 2], [np.nan, np.nan]],
                [[3, 6], [np.nan, np.nan]],
            ],
            [
                [[3, 1], [np.nan, np.nan]],
                [[2, 6], [np.nan, np.nan]],
                [[4, 5], [np.nan, np.nan]],
            ],
            [
                [[6, 1], [np.nan, np.nan]],
                [[2, 3], [np.nan, np.nan]],
                [[5, 4], [np.nan, np.nan]],
            ],
            [
                [[2, 1], [np.nan, np.nan]],
                [[4, 3], [np.nan, np.nan]],
                [[6, 5], [np.nan, np.nan]],
            ],
        ]
    )



    algo_config = {
        "n": num_teams,
        "t": 2 / 3,
        "s": 2,
        "r": 1,
        "p": np.array(
            [
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
        ),
    }

    sol_str = []
    for week in sol:
        week_new = []
        for day in week:
            day_new = []
            for match in day:
                if False in np.isnan(match):
                    day_new.append("vs".join(match.astype(int).astype(str).tolist()))
                else:
                    day_new.append("-")
            week_new.append(day_new)
        sol_str.append(week_new)

    headers = ["Mon", "Fri", "Sat"]
    print(tabulate(sol_str, tablefmt="grid", headers=headers))

    lns = ALNS(algo_config=algo_config, time_out=5, start_solution=sol)
    print(f"Original solution: {compute_profit(sol, lns.p)}")
    lns.check_solution()
    new_sol = lns.run()
    lns.check_solution()
    print(f"LNS solution: {compute_profit(new_sol, lns.p)}")

    sol_str = []
    for week in new_sol:
        week_new = []
        for day in week:
            day_new = []
            for match in day:
                if False in np.isnan(match):
                    day_new.append("vs".join(match.astype(int).astype(str).tolist()))
                else:
                    day_new.append("-")
            week_new.append(day_new)
        sol_str.append(week_new)

    headers = ["Mon", "Fri", "Sat"]
    print(tabulate(sol_str, tablefmt="grid", headers=headers))
