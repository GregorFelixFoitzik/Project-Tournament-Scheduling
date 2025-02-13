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
    insert_games_max_profit_per_week,
    insert_games_random_week,
    random_reorder_weeks,
    select_n_worst_weeks,
    select_random_weeks,
    reorder_week_max_profit,
)
from src.validation import validate
from src.helper import compute_profit, print_solution


class LNS:
    """Class contains the implementation of a LNS.

    Args:
        algo_config (dict[str, Union[int, float, np.ndarray]]): Dictionary containing
            some information about the dataset.
        timeout (float): Timeout for the Metaheuristic
        start_solution (np.ndarray): Start-solution that should be improved.
        runtime_construction: Runtime of the Round Robin Scheduler
    """

    def __init__(
        self,
        algo_config: dict[str, Union[int, float, np.ndarray]],
        timeout: float,
        start_solution: np.ndarray,
        rc: float,
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
        while (
            num_iterations_no_change <= 100
            and sum(os.times()[:2]) + avg_runtime + self.rc < self.timeout
        ):
            t0_iteration = time.time()
            # Destroy and repair the solution
            sol_destroyed, games, weeks_changed = self.destroy(sol=best_solution.copy())
            new_sol = self.repair(
                sol=sol_destroyed,
                games=games,
                weeks_changed=weeks_changed,
                elapsed_time=elapsed_time,
            )
            # Compute the profit of the new solution and solution
            profit_new_sol = compute_profit(
                sol=new_sol, profit=np.array(object=self.p), weeks_between=self.r
            )

            # Check if the new solution is better than the old solution
            if profit_new_sol > profit_best_solution:
                best_solution = new_sol.copy()
                profit_best_solution = profit_new_sol
                num_iterations_no_change = 0
            else:
                num_iterations_no_change += 1

            elapsed_time += time.time() - t0_iteration
            num_iterations += 1
            avg_runtime = elapsed_time / num_iterations

        self.best_solution = best_solution

        return best_solution

    def destroy(self, sol: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Destroy the given solution.

        The selection of the destroy parameter is random.

        Args:
            sol (np.ndarray): The solution that should be destroyed.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing the destroyed
                solution, the games of the destroyed weeks and the week numbers.
        """
        # Randomly choose a destroy parameter
        num_destroy_operators = 2
        destroy_operators = list(range(num_destroy_operators))
        # All destroy parameters are equally distributed
        p = [1 / num_destroy_operators for _ in range(num_destroy_operators)]
        destroy_operator = np.random.choice(a=destroy_operators, size=1, p=p)[0]

        weeks_changed = []

        if destroy_operator == 0:
            # Destroy a random week
            weeks_changed, games = select_random_weeks(
                sol=sol,
                number_of_weeks=np.random.randint(low=2, high=10, size=1)[0],
            )
            sol[weeks_changed] = np.full(shape=games.shape, fill_value=np.nan)
        elif destroy_operator == 1:
            # Destroy the n worst weeks
            worst_weeks, games = select_n_worst_weeks(
                sol=sol,
                n=np.random.randint(low=2, high=10, size=1)[0],
                profits=self.p,
                weeks_between=self.r,
            )

            sol[worst_weeks] = np.full(shape=games.shape, fill_value=np.nan)
            weeks_changed = worst_weeks
        else:
            games = np.array(object=[])
            weeks_changed = np.array(object=[])

        return sol, games[np.argsort(weeks_changed)], np.sort(weeks_changed)

    def repair(
        self,
        sol: np.ndarray,
        games: np.ndarray,
        weeks_changed: np.ndarray,
        elapsed_time: float,
    ) -> np.ndarray:
        """Repair the solution.

        Args:
            sol (np.ndarray): Solution that should be repaired.
            games (np.ndarray): Games that correspond to the destroyed weeks.
            weeks_changed (np.ndarray): Which weeks where destroyed?

        Returns:
            np.ndarray: The repaired solution.
        """
        # Randomly choose a repair parameter
        num_repair_operators = 4
        repair_operators = list(range(num_repair_operators))
        # Only allow the exact solution if there are not so many combinations
        if self.n > 6 or weeks_changed.size > 2:
            p = [
                0 if i == 1 else 1 / (num_repair_operators - 1)
                for i in range(num_repair_operators)
            ]
        else:
            # All destroy parameters are equally distributed
            p = [1 / num_repair_operators for _ in range(num_repair_operators)]
        if elapsed_time > 25:
            p = [
                0 if i == 1 or i == 2 else 1 / (num_repair_operators - 2)
                for i in range(num_repair_operators)
            ]

        repair_operator = np.random.choice(a=repair_operators, size=1, p=p)[0]
        repair_operator = 2

        if repair_operator == 0:
            # Random fill
            for i, week_changed in enumerate(iterable=weeks_changed):
                week_new = insert_games_random_week(
                    sol=sol,
                    games_week=games[i],
                    week_changed=week_changed,
                    number_of_teams=self.n,
                    t=self.t,
                )
                sol[week_changed] = week_new
        elif repair_operator == 1:
            games_old = games.copy()
            # Extract all games
            games = games[np.logical_not(np.isnan(games))]
            games = games.reshape(int(games.shape[0] / 2), 2).astype(dtype=int)

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
                    t=self.t,
                )
                sol = max_sol.copy()
        elif repair_operator == 2:
            for i, week_changed in enumerate(iterable=weeks_changed):
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
        elif repair_operator == 3:
            # Random re-ordering of destroyed weeks
            sol = random_reorder_weeks(
                sol=sol, games=games, weeks_changed=weeks_changed
            )

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
