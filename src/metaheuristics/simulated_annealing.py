"""This file contains the implementation of a Tabu-Search."""

# Standard library
from select import epoll
import time
import random
import itertools

from typing import Union
from datetime import timedelta

# Third party library
from attr import asdict
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
from src.neighborhoods import insert_games_random_week, select_random_weeks


class SimulatedAnnealing:
    def __init__(
        self,
        algo_config: dict[str, Union[int, float, np.ndarray]],
        time_out: int,
        start_solution: np.ndarray,
        temperature: float,
        alpha: float,
        epsilon: float,
        neighborhood: str,
    ) -> None:
        self.n = algo_config["n"]  # int
        self.t = algo_config["t"]  # float
        self.s = algo_config["s"]  # int
        self.r = algo_config["r"]  # int
        # Has shape (3, n, n), so self.p[0] is the profit for monday
        self.p = np.array(algo_config["p"]).reshape((3, self.n, self.n))

        self.time_out = time_out
        self.sol = start_solution
        self.best_solution = start_solution

        self.all_teams = range(1, self.n + 1)
        self.neighborhood = neighborhood
        self.temperature = temperature
        self.alpha = alpha
        self.epsilon = epsilon
        self.neighborhoods = {
            "random_swap_within_week": self.random_swap_within_week,
            "select_worst_n_weeks": self.select_worst_n_weeks,
        }

    def run(self) -> np.ndarray:
        """
        Execute algorithm

        Args:
           None

        Returns:
            None
        """
        sol = self.sol.copy()
        best_solution = sol.copy()

        profit_best_solution = compute_profit(best_solution, self.p, self.r)

        t0 = time.time()
        while self.temperature >= self.epsilon and time.time() - t0 < self.time_out:
            new_sol = self.neighborhoods[self.neighborhood](sol)
            profit_new_sol = compute_profit(
                sol=new_sol, profit=np.array(object=self.p), weeks_between=self.r
            )
            profit_sol = compute_profit(
                sol=sol, profit=np.array(object=self.p), weeks_between=self.r
            )
            if (
                profit_new_sol > profit_sol
                and np.exp(-(profit_new_sol - profit_sol) / self.temperature)
                > np.random.uniform()
            ):
                sol = new_sol

            if profit_new_sol > profit_best_solution:
                profit_best_solution = profit_new_sol
                best_solution = new_sol.copy()

            self.temperature *= self.alpha

        self.best_solution = best_solution

        return best_solution

    def random_swap_within_week(self, sol: np.ndarray) -> np.ndarray:
        # Extract one random week
        random_week, games_week = select_random_weeks(sol=sol, number_of_weeks=1)
        random_week = random_week[0]
        games_week = games_week[0]

        week_new = insert_games_random_week(
            sol=sol,
            games_week=games_week,
            week_changed=random_week,
            number_of_teams=self.n,
        )

        new_sol = sol.copy()
        new_sol[random_week] = week_new

        return new_sol

    def select_worst_n_weeks(self, sol: np.ndarray) -> np.ndarray:
        # Extract the two worst weeks
        worst_weeks, games = select_n_worst_weeks(
            sol=sol, n=2, profits=self.p, weeks_between=self.r
        )

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
                weeks_changed=worst_weeks,
                profits=self.p,
                num_teams=self.n,
                weeks_between=self.r,
            )
            sol = max_sol.copy()
            try:
                validate(sol, self.n)
            except Exception:
                print("asd")

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
#     num_teams = 6

#     sol = np.array(
#         [
#             [
#                 [[5, 1], [np.nan, np.nan]],
#                 [[3, 6], [2, 4]],
#                 [[np.nan, np.nan], [np.nan, np.nan]],
#             ],
#             [
#                 [[6, 2], [np.nan, np.nan]],
#                 [[1, 4], [3, 5]],
#                 [[np.nan, np.nan], [np.nan, np.nan]],
#             ],
#             [
#                 [[5, 4], [np.nan, np.nan]],
#                 [[1, 6], [np.nan, np.nan]],
#                 [[3, 2], [np.nan, np.nan]],
#             ],
#             [
#                 [[6, 4], [np.nan, np.nan]],
#                 [[1, 3], [np.nan, np.nan]],
#                 [[5, 2], [np.nan, np.nan]],
#             ],
#             [
#                 [[1, 2], [np.nan, np.nan]],
#                 [[3, 4], [5, 6]],
#                 [[np.nan, np.nan], [np.nan, np.nan]],
#             ],
#             [
#                 [[6, 3], [np.nan, np.nan]],
#                 [[4, 1], [np.nan, np.nan]],
#                 [[2, 5], [np.nan, np.nan]],
#             ],
#             [
#                 [[3, 1], [np.nan, np.nan]],
#                 [[4, 2], [np.nan, np.nan]],
#                 [[6, 5], [np.nan, np.nan]],
#             ],
#             [
#                 [[1, 5], [np.nan, np.nan]],
#                 [[4, 3], [2, 6]],
#                 [[np.nan, np.nan], [np.nan, np.nan]],
#             ],
#             [
#                 [[4, 5], [np.nan, np.nan]],
#                 [[2, 3], [np.nan, np.nan]],
#                 [[6, 1], [np.nan, np.nan]],
#             ],
#             [
#                 [[4, 6], [np.nan, np.nan]],
#                 [[5, 3], [2, 1]],
#                 [[np.nan, np.nan], [np.nan, np.nan]],
#             ],
#         ]
#     )
#     # sol = np.array(
#     #     [
#     #         [
#     #             [[5, 1], [np.nan, np.nan]],
#     #             [[3, 2], [np.nan, np.nan]],
#     #             [[6, 4], [np.nan, np.nan]],
#     #         ],
#     #         [
#     #             [[6, 2], [np.nan, np.nan]],
#     #             [[1, 4], [np.nan, np.nan]],
#     #             [[3, 5], [np.nan, np.nan]],
#     #         ],
#     #         [
#     #             [[1, 6], [np.nan, np.nan]],
#     #             [[2, 4], [np.nan, np.nan]],
#     #             [[5, 3], [np.nan, np.nan]],
#     #         ],
#     #         [
#     #             [[1, 3], [np.nan, np.nan]],
#     #             [[5, 2], [np.nan, np.nan]],
#     #             [[4, 6], [np.nan, np.nan]],
#     #         ],
#     #         [
#     #             [[1, 2], [np.nan, np.nan]],
#     #             [[3, 4], [np.nan, np.nan]],
#     #             [[5, 6], [np.nan, np.nan]],
#     #         ],
#     #         [
#     #             [[4, 1], [np.nan, np.nan]],
#     #             [[2, 5], [np.nan, np.nan]],
#     #             [[6, 3], [np.nan, np.nan]],
#     #         ],
#     #         [
#     #             [[1, 5], [np.nan, np.nan]],
#     #             [[4, 2], [np.nan, np.nan]],
#     #             [[3, 6], [np.nan, np.nan]],
#     #         ],
#     #         [
#     #             [[3, 1], [np.nan, np.nan]],
#     #             [[2, 6], [np.nan, np.nan]],
#     #             [[4, 5], [np.nan, np.nan]],
#     #         ],
#     #         [
#     #             [[6, 1], [np.nan, np.nan]],
#     #             [[2, 3], [np.nan, np.nan]],
#     #             [[5, 4], [np.nan, np.nan]],
#     #         ],
#     #         [
#     #             [[2, 1], [np.nan, np.nan]],
#     #             [[4, 3], [np.nan, np.nan]],
#     #             [[6, 5], [np.nan, np.nan]],
#     #         ],
#     #     ]
#     # )

#     algo_config = {
#         "n": num_teams,
#         "t": 2 / 3,
#         "s": 2,
#         "r": 3,
#         "p": np.array(
#             [
#                 -1.0,
#                 15.0,
#                 20.0,
#                 16.0,
#                 28.0,
#                 9.0,
#                 13.0,
#                 -1.0,
#                 8.0,
#                 12.0,
#                 10.0,
#                 12.0,
#                 19.0,
#                 10.0,
#                 -1.0,
#                 14.0,
#                 15.0,
#                 13.0,
#                 18.0,
#                 14.0,
#                 12.0,
#                 -1.0,
#                 13.0,
#                 16.0,
#                 25.0,
#                 15.0,
#                 13.0,
#                 17.0,
#                 -1.0,
#                 11.0,
#                 10.0,
#                 14.0,
#                 23.0,
#                 21.0,
#                 10.0,
#                 -1.0,
#                 -1.0,
#                 8.0,
#                 10.0,
#                 9.0,
#                 5.0,
#                 9.0,
#                 6.0,
#                 -1.0,
#                 5.0,
#                 11.0,
#                 7.0,
#                 4.0,
#                 8.0,
#                 4.0,
#                 -1.0,
#                 12.0,
#                 8.0,
#                 6.0,
#                 10.0,
#                 7.0,
#                 4.0,
#                 -1.0,
#                 4.0,
#                 2.0,
#                 10.0,
#                 7.0,
#                 5.0,
#                 3.0,
#                 -1.0,
#                 10.0,
#                 4.0,
#                 8.0,
#                 2.0,
#                 7.0,
#                 1.0,
#                 -1.0,
#                 -1.0,
#                 2.0,
#                 3.0,
#                 5.0,
#                 3.0,
#                 5.0,
#                 1.0,
#                 -1.0,
#                 3.0,
#                 2.0,
#                 9.0,
#                 10.0,
#                 7.0,
#                 8.0,
#                 -1.0,
#                 4.0,
#                 5.0,
#                 2.0,
#                 6.0,
#                 7.0,
#                 5.0,
#                 -1.0,
#                 2.0,
#                 1.0,
#                 6.0,
#                 8.0,
#                 5.0,
#                 4.0,
#                 -1.0,
#                 7.0,
#                 8.0,
#                 3.0,
#                 2.0,
#                 7.0,
#                 4.0,
#                 -1.0,
#             ]
#         ),
#     }

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

#     simulated_annealing = SimulatedAnnealing(
#         algo_config=algo_config,
#         time_out=30,
#         start_solution=sol,
#         alpha=0.95,
#         temperature=10000,
#         epsilon=0.001,
#         neighborhood="select_worst_n_weeks",
#     )
#     print(
#         f"Original solution: {compute_profit(sol, simulated_annealing.p, algo_config['r'])}"
#     )
#     simulated_annealing.check_solution()
#     t0 = time.time()
#     new_sol = simulated_annealing.run()
#     simulated_annealing.check_solution()
#     print(
#         f"Simulated annealing solution ({np.round(time.time()- t0, 5)}): {compute_profit(new_sol, simulated_annealing.p, algo_config['r'])}"
#     )

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
