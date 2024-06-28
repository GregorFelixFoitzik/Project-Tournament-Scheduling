# Standard library
import os

# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
import time

from typing import Union
from datetime import timedelta

# Third party library
import numpy as np

# Project specific library
from src.validation import validate
from src.neighborhoods import (
    insert_games_random_week,
    reorder_week_max_profit,
    select_n_worst_weeks,
    select_random_weeks,
)
from src.helper import compute_profit, print_solution


class TabuSearch:
    """Class contains the implementation of a Tabu-Search algorithm.

    Args:
        algo_config (dict[str, Union[int, float, np.ndarray]]): Dictionary containing
            some information about the dataset.
        timeout (float): Timeout for the Metaheuristic
        start_solution (np.ndarray): Start-solution that should be improved.
        runtime_construction: Runtime of the Round Robin Scheduler
        max_size_tabu_list (int): Max size of the Tabu-List.
    """

    def __init__(
        self,
        algo_config: dict[str, Union[int, float, np.ndarray]],
        timeout: float,
        start_solution: np.ndarray,
        rc: float,
        max_size_tabu_list: int,
        neighborhood: str,
    ) -> None:
        self.n = int(algo_config["n"])
        self.t = float(algo_config["t"])
        self.s = int(algo_config["s"])
        self.r = int(algo_config["r"])
        # Has shape (3, n, n), so self.p[0] is the profit for monday
        self.p = np.array(object=algo_config["p"]).reshape((3, self.n, self.n))

        self.timeout = timeout
        self.sol = start_solution
        self.best_solution = start_solution
        self.rc = rc

        self.all_teams = range(1, self.n + 1)
        self.max_size_tabu_list = max_size_tabu_list
        self.neighborhood = neighborhood
        self.neighborhoods = {
            "random_swap_within_week": self.random_swap_within_week,
            "select_worst_n_weeks": self.select_worst_n_weeks,
        }

    def run(self) -> np.ndarray:
        """Execute the metaheuristic.

        Returns:
            np.ndarray: The improved solution.
        """
        # Set the start sol as best solution
        start_solution = self.sol.copy()
        best_solution = start_solution
        profit_best_solution = compute_profit(
            sol=best_solution, profit=self.p, weeks_between=self.r
        )

        num_iterations_no_change = 0

        t0 = time.time()
        elapsed_time = 0
        num_iterations = 0
        avg_runtime = 0
        tabu_list = []
        while (
            num_iterations_no_change <= 100
            and sum(os.times()[:2]) + avg_runtime + self.rc < self.timeout
        ):
            t0_iteration = time.time()
            new_sol, weeks_changed = self.neighborhoods[self.neighborhood](
                best_solution
            )

            profit_new_sol = compute_profit(
                sol=new_sol, profit=np.array(object=self.p), weeks_between=self.r
            )

            if profit_new_sol > profit_best_solution:
                best_solution = new_sol.copy()
                profit_best_solution = profit_new_sol
                num_iterations_no_change = 0
            else:
                num_iterations_no_change += 1

            tabu_list += weeks_changed.tolist()
            if len(tabu_list) > self.max_size_tabu_list:
                for _ in range(
                    len(tabu_list) - self.max_size_tabu_list,
                ):
                    tabu_list.pop(0)
            # print(len(tabu_list))

            elapsed_time += time.time() - t0_iteration
            num_iterations += 1
            avg_runtime = elapsed_time / num_iterations

        self.best_solution = best_solution

        return best_solution

    def random_swap_within_week(self, sol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Swap the games within a week randomly.

        Args:
            sol (np.ndarray): Solution that should be changed.

        Returns:
            np.ndarray: Updated solution.
        """
        # Extract one random week
        random_weeks, games_week = select_random_weeks(
            sol=sol, number_of_weeks=np.random.randint(low=1, high=10, size=1)[0]
        )

        new_sol = sol.copy()
        for week in random_weeks:
            week_new = insert_games_random_week(
                sol=sol,
                games_week=games_week,
                week_changed=week,
                number_of_teams=self.n,
                t=self.t,
            )
            new_sol[week] = week_new

        return new_sol, random_weeks

    def select_worst_n_weeks(self, sol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Select the $n$-worst weeks and reorder the weeks, so profit is maximized.

        Args:
            sol (np.ndarray): Solution that should be changed.

        Returns:
            np.ndarray: Updated solution.
        """
        # Extract the two worst weeks
        worst_weeks, games = select_n_worst_weeks(
            sol=sol,
            n=np.random.randint(low=2, high=10, size=1)[0],
            profits=self.p,
            weeks_between=self.r,
        )

        sol[worst_weeks] = np.full(shape=games.shape, fill_value=np.nan)
        weeks_changed = worst_weeks

        for i, week_changed in enumerate(iterable=worst_weeks):
            week_updated = reorder_week_max_profit(
                sol=sol,
                profits=self.p,
                games=games[i],
                num_teams=self.n,
                t=self.t,
                current_week=week_changed,
                weeks_between=self.r,
            )

            sol[week_changed] = week_updated

        try:
            validate(sol, self.n)
        except Exception:
            print("sd")

        return sol, weeks_changed

    def check_solution(self) -> bool:
        """Check if the solutionis valued

        Returns:
            bool: True fi solution is valid, otherise an Assertion-Errror is raised.
        """
        validation = validate(sol=self.sol, num_teams=self.n)
        assert validation is True

        return True

    def execute_cmd(self):
        # https://stackoverflow.com/a/61713634 28.05.2024
        start = time.perf_counter()
        solution_set = self.run()
        runtime = timedelta(seconds=time.perf_counter() - start)

        print_solution(runtime, solution_set)
