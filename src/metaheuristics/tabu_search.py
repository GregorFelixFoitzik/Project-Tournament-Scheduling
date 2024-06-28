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
from src.neighborhoods import (
    insert_games_random_week,
)
from src.validation import validate
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
        self.rc=rc 

        self.all_teams = range(1, self.n + 1)
        self.max_size_tabu_list = max_size_tabu_list

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
            # Randomly select n weeks
            possible_weeks = np.setdiff1d(
                ar1=list(range(best_solution.shape[0])), ar2=tabu_list
            )
            number_of_weeks = min(4, possible_weeks.shape[0])
            weeks_changed = np.random.choice(
                a=possible_weeks,
                size=number_of_weeks,
                replace=False,
            )
            games = best_solution[weeks_changed].copy()

            new_sol = best_solution.copy()

            for i, week_changed in enumerate(iterable=weeks_changed):
                week_new = insert_games_random_week(
                    sol=new_sol,
                    games_week=games[i],
                    week_changed=week_changed,
                    number_of_teams=self.n,
                    t=self.t,
                )
                new_sol[week_changed] = week_new

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
