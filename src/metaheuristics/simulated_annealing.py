"""This file contains the implementation of a Tabu-Search."""

# Standard library
import os
import time

from typing import Union
from datetime import timedelta

# Third party library
import numpy as np

# Project specific library
from src.neighborhoods import (
    insert_games_random_week,
    reorder_week_max_profit,
    select_n_worst_weeks,
    select_random_weeks,
)
from src.validation import validate
from src.helper import compute_profit, print_solution


class SimulatedAnnealing:
    """Class contains the implementation of a Simualted-Annealing algorithm.

    Args:
        algo_config (dict[str, Union[int, float, np.ndarray]]): Dictionary containing
            some information about the dataset.
        timeout (float): Timeout for the Metaheuristic
        start_solution (np.ndarray): Start-solution that should be improved.
        runtime_construction: Runtime of the Round Robin Scheduler
        temperature (float): Temeprature value for the simulated-annealing part.
        alpha (float): Alpha for the simulated-annealing part.
        epsilon (float): Epsilon for the simulated-annelaing part.
        neighborhood (str): Which neighborhood should be used?
    """

    def __init__(
        self,
        algo_config: dict[str, Union[int, float, np.ndarray]],
        timeout: float,
        start_solution: np.ndarray,
        rc: float, 
        temperature: float,
        alpha: float,
        epsilon: float,
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
        self.rc=rc

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
        """Execute the metaheuristic.

        Returns:
            np.ndarray: The improved solution.
        """
        # Set the start sol as best solution
        sol = self.sol.copy()
        best_solution = sol.copy()

        profit_best_solution = compute_profit(
            sol=best_solution, profit=self.p, weeks_between=self.r
        )

        t0 = time.time()

        elapsed_time = 0
        num_iterations = 0
        avg_runtime = 0
        while (self.temperature >= self.epsilon and sum(os.times()[:2]) + avg_runtime + self.rc < self.timeout):
            t0_iteration = time.time()
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

            # Check if the new solution is better than the old solution
            if profit_sol > profit_best_solution:
                best_solution = sol.copy()
                profit_best_solution = profit_sol

            self.temperature *= self.alpha

            elapsed_time += time.time() - t0_iteration
            num_iterations += 1
            avg_runtime = elapsed_time / num_iterations

        self.best_solution = best_solution

        return best_solution

    def random_swap_within_week(self, sol: np.ndarray) -> np.ndarray:
        """Swap the games within a week randomly.

        Args:
            sol (np.ndarray): Solution that should be changed.

        Returns:
            np.ndarray: Updated solution.
        """
        # Extract one random week
        random_week, games_week = select_random_weeks(sol=sol, number_of_weeks=1)
        random_week = random_week[0]
        games_week = games_week[0]

        week_new = insert_games_random_week(
            sol=sol,
            games_week=games_week,
            week_changed=random_week,
            number_of_teams=self.n,
            t=self.t,
        )

        new_sol = sol.copy()
        new_sol[random_week] = week_new

        return new_sol

    def select_worst_n_weeks(self, sol: np.ndarray) -> np.ndarray:
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
            print('sd')

        return sol

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
