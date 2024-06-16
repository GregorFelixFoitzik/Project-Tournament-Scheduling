# Standard library
import copy
import time
import random
import itertools

from datetime import timedelta
import trace
from typing import Union

# Third party library
from attr import asdict
import numpy as np

from tabulate import tabulate


# Project specific library
from validation import every_team_every_week, validate
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
        profit_best_solution = compute_profit(best_solution, self.p, self.r)

        num_iterations_no_change = 0

        t0 = time.time()
        while num_iterations_no_change <= 100 and time.time() - t0 < self.time_out:
            sol_destroyed, games, weeks_changed = self.destroy(best_solution.copy())
            new_sol = self.repair(sol_destroyed, games, weeks_changed)
            profit_new_sol = compute_profit(
                sol=new_sol, profit=np.array(object=self.p), weeks_between=self.r
            )

            if profit_new_sol > profit_best_solution:
                best_solution = new_sol.copy()
                profit_best_solution = profit_new_sol
                num_iterations_no_change = 0
            else:
                num_iterations_no_change += 1

        self.best_solution = best_solution

        return best_solution

    def destroy(self, sol: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Randomly choose a destroy parameter
        num_destroy_operators = 2
        destroy_operators = list(range(num_destroy_operators))
        weights = [100] * num_destroy_operators
        destroy_operator = random.choices(destroy_operators, weights=weights)[0]

        weeks_changed = []

        if destroy_operator == 0:
            # Destroy 2 weeks randomly
            weeks_changed = np.random.choice(
                list(range(sol.shape[0])), size=2, replace=False
            )
            games = sol[weeks_changed].copy()
            sol[weeks_changed] = np.full(games.shape, np.nan)
        elif destroy_operator == 1:
            # Destry the two worst weeks
            week_profits = get_profits_per_week(sol, self.p, self.r)

            worst_weeks = np.argsort(week_profits)[:2]
            games = sol[worst_weeks].copy()

            sol[worst_weeks] = np.full(games.shape, np.nan)
            weeks_changed = worst_weeks
        else:
            games = np.array([])
            weeks_changed = np.array([])

        return sol, games, weeks_changed

    def repair(self, sol: np.ndarray, games: np.ndarray, weeks_changed: np.ndarray):
        # Randomly choose a repai parameter
        num_repair_operators = 2
        repair_operators = list(range(num_repair_operators))
        weights = [100] * num_repair_operators
        repair_operator = random.choices(repair_operators, weights=weights)[0]

        if repair_operator == 0:
            # Random fill
            for i, week_changed in enumerate(weeks_changed):
                games_week = games[i]
                games_unique = np.unique(games_week, axis=1)
                games_unique = games_unique[np.logical_not(np.isnan(games_unique))]
                games_unique = games_unique.reshape(int(games_unique.shape[0] / 2), 2)

                # Which teams have to play on monday and how does the new week look like?
                teams_play_on_monday = np.unique(sol[:, 0])[:-1]
                week_new = np.full(games_week.shape, np.nan)

                # Set the monday game
                if np.setdiff1d(range(1, self.n+1), teams_play_on_monday).size == 0:
                    monday_game_idx = np.random.choice(games_unique.shape[0])
                    week_new[0][0] = games_unique[monday_game_idx]
                else:
                    monday_game_idx = np.where(games_unique == np.setdiff1d(range(1, self.n+1), teams_play_on_monday))[0]
                    week_new[0][0] = games_unique[monday_game_idx]
                
                # Randomly distribute the remaiing games
                remaining_games = games_unique[games_unique != games_unique[monday_game_idx]]
                remaining_games = remaining_games.reshape(int(remaining_games.shape[0] / 2), 2)

                random_choice = np.random.choice([1, 2], size=remaining_games.shape[0])

                for game_idx, day_choice in enumerate(random_choice):
                    week_new[day_choice][np.where(np.isnan(week_new[day_choice]))[0][0]] = remaining_games[game_idx]

                sol[week_changed] = week_new
        elif repair_operator == 1:
            games_old = games.copy()
            # Extract all games
            games = games[np.logical_not(np.isnan(games))]
            games = games.reshape(int(games.shape[0] / 2), 2).astype(int)
            teams = np.unique(games)

            games_encoded = [i for i in range(games.shape[0])]

            # Iterate over the possible combinations extract those, where each
            #   team is present
            possible_combinations = []
            for num_repetitions in range(int(self.n / 2), int(self.n * self.t)):
                # Source: https://stackoverflow.com/a/5898031, accessed 11th June
                combinations = itertools.permutations(games_encoded, num_repetitions)
                possible_combinations_tmp = []
                possible_combinations_tmp_idx = []
                # Iterate over each game combination and check some of th constraints
                for combination_idx in combinations:
                    combination = games[list(combination_idx)]
                    # Check if all teams play during that week
                    if np.unique(ar=combination).size != len(list(self.all_teams)):
                        continue
                    if np.all(np.unique(combination) != list(self.all_teams)):
                        continue

                    # possible_combinations_tmp.append(np.array(combination))
                    possible_combinations_tmp_idx.append(combination_idx)

                possible_combinations_tmp = np.array(possible_combinations_tmp)

                possible_weekly_combinations = []
                # Create all possible weekly combinations
                weekly_combinations = np.array(
                    list(
                        itertools.permutations(
                            possible_combinations_tmp_idx, weeks_changed.size
                        )
                    )
                )
                # Go over the weekly-combinations and drop all duplicate games
                for weekly_combination_idx in weekly_combinations:
                    weekly_combination = games[weekly_combination_idx]
                    weekly_combination_games = weekly_combination.reshape(
                        int(weekly_combination.size / 2), 2
                    )
                    # Drop all duplicate games
                    if (
                        weekly_combination_games.shape
                        != np.unique(weekly_combination_games, axis=0).shape
                    ):
                        continue
                    possible_weekly_combinations.append(weekly_combination)

                # Get max profit when inserted in solution for each combination
                max_profit = 0
                max_sol = sol.copy()
                for weekly_combination in possible_weekly_combinations:
                    sol_new = sol.copy()
                    # Insert each weekly-combination into the solution
                    for i, week in enumerate(weekly_combination):
                        week_new = np.full(games_old[0].shape, np.nan)
                        week_new[0][0] = week[0]
                        for game in week[1:]:
                            games_position = (
                                np.argmax(self.p[1:, game[0] - 1, game[1] - 1]) + 1
                            )
                            week_new[games_position][
                                np.where(np.isnan(week_new[games_position]))[0][0]
                            ] = game
                        sol_new[weeks_changed[i]] = week_new

                    # Check if the solution is valid or not
                    try:
                        validate(sol_new, self.n)
                    except AssertionError:
                        continue

                    # If the solution is valid: Does the solution give a higher profit?
                    if compute_profit(sol_new, self.p, self.r) > max_profit:
                        max_profit = compute_profit(sol_new, self.p, self.r)
                        max_sol = sol_new.copy()
                sol = max_sol.copy()
                try:
                    validate(sol, self.n)
                except Exception:
                    print("asd")
        self.sol = sol
        return sol

    def check_solution(self):
        """
        not feasible returns none
        is feasible returns np.array
        """
        validation = validate(self.sol, self.n)
        assert validation == True

    def execute_cmd(self):
        # https://stackoverflow.com/a/61713634 28.05.2024
        start = time.perf_counter()
        solution_set = self.run()
        runtime = timedelta(seconds=time.perf_counter() - start)

        print_solution(runtime, solution_set)


if __name__ == "__main__":
    num_teams = 6

    sol = np.array(
        [
            [
                [[5, 1], [np.nan, np.nan]],
                [[3, 6], [2, 4]],
                [[np.nan, np.nan], [np.nan, np.nan]],
            ],
            [
                [[6, 2], [np.nan, np.nan]],
                [[1, 4], [3, 5]],
                [[np.nan, np.nan], [np.nan, np.nan]],
            ],
            [
                [[5, 4], [np.nan, np.nan]],
                [[1, 6], [np.nan, np.nan]],
                [[3, 2], [np.nan, np.nan]],
            ],
            [
                [[6, 4], [np.nan, np.nan]],
                [[1, 3], [np.nan, np.nan]],
                [[5, 2], [np.nan, np.nan]],
            ],
            [
                [[1, 2], [np.nan, np.nan]],
                [[3, 4], [5, 6]],
                [[np.nan, np.nan], [np.nan, np.nan]],
            ],
            [
                [[6, 3], [np.nan, np.nan]],
                [[4, 1], [np.nan, np.nan]],
                [[2, 5], [np.nan, np.nan]],
            ],
            [
                [[3, 1], [np.nan, np.nan]],
                [[4, 2], [np.nan, np.nan]],
                [[6, 5], [np.nan, np.nan]],
            ],
            [
                [[1, 5], [np.nan, np.nan]],
                [[4, 3], [2, 6]],
                [[np.nan, np.nan], [np.nan, np.nan]],
            ],
            [
                [[4, 5], [np.nan, np.nan]],
                [[2, 3], [np.nan, np.nan]],
                [[6, 1], [np.nan, np.nan]],
            ],
            [
                [[4, 6], [np.nan, np.nan]],
                [[5, 3], [2, 1]],
                [[np.nan, np.nan], [np.nan, np.nan]],
            ],
        ]
    )
    # sol = np.array(
    #     [
    #         [
    #             [[5, 1], [np.nan, np.nan]],
    #             [[3, 2], [np.nan, np.nan]],
    #             [[6, 4], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[6, 2], [np.nan, np.nan]],
    #             [[1, 4], [np.nan, np.nan]],
    #             [[3, 5], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[1, 6], [np.nan, np.nan]],
    #             [[2, 4], [np.nan, np.nan]],
    #             [[5, 3], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[1, 3], [np.nan, np.nan]],
    #             [[5, 2], [np.nan, np.nan]],
    #             [[4, 6], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[1, 2], [np.nan, np.nan]],
    #             [[3, 4], [np.nan, np.nan]],
    #             [[5, 6], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[4, 1], [np.nan, np.nan]],
    #             [[2, 5], [np.nan, np.nan]],
    #             [[6, 3], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[1, 5], [np.nan, np.nan]],
    #             [[4, 2], [np.nan, np.nan]],
    #             [[3, 6], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[3, 1], [np.nan, np.nan]],
    #             [[2, 6], [np.nan, np.nan]],
    #             [[4, 5], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[6, 1], [np.nan, np.nan]],
    #             [[2, 3], [np.nan, np.nan]],
    #             [[5, 4], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[2, 1], [np.nan, np.nan]],
    #             [[4, 3], [np.nan, np.nan]],
    #             [[6, 5], [np.nan, np.nan]],
    #         ],
    #     ]
    # )

    algo_config = {
        "n": num_teams,
        "t": 2 / 3,
        "s": 2,
        "r": 3,
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
            for game in day:
                if False in np.isnan(game):
                    day_new.append("vs".join(game.astype(int).astype(str).tolist()))
                else:
                    day_new.append("-")
            week_new.append(day_new)
        sol_str.append(week_new)

    headers = ["Mon", "Fri", "Sat"]
    print(tabulate(sol_str, tablefmt="grid", headers=headers))

    lns = ALNS(algo_config=algo_config, time_out=30, start_solution=sol)
    print(f"Original solution: {compute_profit(sol, lns.p, algo_config['r'])}")
    lns.check_solution()
    new_sol = lns.run()
    lns.check_solution()
    print(f"LNS solution: {compute_profit(new_sol, lns.p, algo_config['r'])}")

    sol_str = []
    for week in new_sol:
        week_new = []
        for day in week:
            day_new = []
            for game in day:
                if False in np.isnan(game):
                    day_new.append("vs".join(game.astype(int).astype(str).tolist()))
                else:
                    day_new.append("-")
            week_new.append(day_new)
        sol_str.append(week_new)

    headers = ["Mon", "Fri", "Sat"]
    print(tabulate(sol_str, tablefmt="grid", headers=headers))
