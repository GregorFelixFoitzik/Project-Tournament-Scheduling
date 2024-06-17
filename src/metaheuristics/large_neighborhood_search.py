# Standard library
import copy
import time
import random
import itertools

from datetime import timedelta
import trace
from typing import Union

# Third party library
import numpy as np

from tabulate import tabulate


# Project specific library
from src.neighborhoods import (
    insert_games_max_profit_per_week,
    insert_games_random_week,
    select_n_worst_weeks,
    select_random_weeks,
)
from src.validation import validate
from src.helper import compute_profit, print_solution


class LNS:
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
            weeks_changed, games = select_random_weeks(sol=sol, number_of_weeks=2)
            sol[weeks_changed] = np.full(games.shape, np.nan)
        elif destroy_operator == 1:
            # Destroy the two worst weeks
            worst_weeks, games = select_n_worst_weeks(
                sol=sol, n=2, profits=self.p, weeks_between=self.r
            )

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
                week_new = insert_games_random_week(
                    sol=sol,
                    games_week=games[i],
                    week_changed=week_changed,
                    number_of_teams=self.n,
                )
                sol[week_changed] = week_new
        elif repair_operator == 1:
            games_old = games.copy()
            # Extract all games
            games = games[np.logical_not(np.isnan(games))]
            games = games.reshape(int(games.shape[0] / 2), 2).astype(int)

            games_encoded = [i for i in range(games.shape[0])]

            # Iterate over the possible combinations extract those, where each
            #   team is present
            for num_repetitions in range(int(self.n / 2), int(self.n * self.t)):
                max_sol = insert_games_max_profit_per_week(
                    sol=sol,
                    games_old=games_old,
                    games_encoded=games_encoded,
                    num_repetitions=num_repetitions,
                    games=games,
                    all_teams=list(self.all_teams),
                    weeks_changed=weeks_changed,
                    profits=self.p,
                    num_teams=self.n,
                    weeks_between=self.r,
                )
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


# if __name__ == "__main__":
    

#     sol_str = []
#     for week in sol:
#         week_new = []
#         for day in week:
#             day_new = []
#             for game in day:
#                 if False in np.isnan(game):
#                     day_new.append("vs".join(game.astype(int).astype(str).tolist()))
#                 else:
#                     day_new.append("-")
#             week_new.append(day_new)
#         sol_str.append(week_new)

#     headers = ["Mon", "Fri", "Sat"]
#     print(tabulate(sol_str, tablefmt="grid", headers=headers))

#     lns = ALNS(algo_config=algo_config, time_out=30, start_solution=sol)
#     print(f"Original solution: {compute_profit(sol, lns.p, algo_config['r'])}")
#     lns.check_solution()
#     t0 = time.time()
#     new_sol = lns.run()
#     lns.check_solution()
#     print(f"LNS solution {np.round(time.time() - t0, 5)}s: {compute_profit(new_sol, lns.p, algo_config['r'])}")

#     sol_str = []
#     for week in new_sol:
#         week_new = []
#         for day in week:
#             day_new = []
#             for game in day:
#                 if False in np.isnan(game):
#                     day_new.append("vs".join(game.astype(int).astype(str).tolist()))
#                 else:
#                     day_new.append("-")
#             week_new.append(day_new)
#         sol_str.append(week_new)

#     headers = ["Mon", "Fri", "Sat"]
#     print(tabulate(sol_str, tablefmt="grid", headers=headers))
