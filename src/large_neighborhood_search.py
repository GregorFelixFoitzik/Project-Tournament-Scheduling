# Standard library
import time
import random

from datetime import timedelta
from typing import Union

# Third party library
import numpy as np

# Project specific library
from helper import print_solution


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
        self.p = algo_config["p"]  # np.array

        self.time_out = time_out
        self.start_solution = start_solution

    def run(self) -> np.ndarray:
        """
        Execute algorithm

        Args:
           None

        Returns:
            None
        """
        start_solution = self.start_solution.copy()
        best_solution = start_solution

        t0 = time.time()
        while time.time() - t0 < self.time_out:
            sol_destroyed, matches, weeks_changed = self.destroy(sol)
            sol_tmp = self.repair(sol_destroyed, matches, weeks_changed)

    def destroy(self, sol: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[int]]:
        # Randomly choose a destroy parameter
        num_destroy_operators = 1
        destroy_operators = list(range(num_destroy_operators))
        weights = [100] * num_destroy_operators
        destroy_operator = random.choices(destroy_operators, weights=weights)[0]

        weeks_changed = []

        if destroy_operator == 0:
            # Destroy one week randomly
            week_to_destroy = np.random.choice(list(range(sol.shape[0])))
            matches = sol[week_to_destroy].copy()
            sol[week_to_destroy] = np.full(matches.shape, np.nan)
            weeks_changed = [week_to_destroy]
        else:
            matches = np.array([])

        return sol, matches, weeks_changed

    def repair(self, sol: np.ndarray, matches: np.ndarray, weeks_changed: list[int]):
        # Randomly choose a repai parameter
        num_destroy_operators = 1
        repair_operators = list(range(num_destroy_operators))
        weights = [100] * num_destroy_operators
        repair_operator = random.choices(repair_operators, weights=weights)[0]

        if repair_operator == 0:
            # Random fill
            for i, week_changed in enumerate(weeks_changed):
                new_order = np.full(sol.shape[1:], np.nan, dtype=str).tolist()

                matches_unique = np.unique(matches[i])
                matches_unique = matches_unique[matches_unique != 'nan']
                monday_game = random.choice(matches_unique)
                new_order[0][0] = monday_game
                new_order = np.array(new_order)

                matches_unique = matches_unique[matches_unique!=monday_game]

                sol[week_changed] = new_order


            print("asd")

    def check_solution():
        """
        not feasible returns none
        is feasible returns np.array
        """
        pass

    def execute_cmd(self):
        # https://stackoverflow.com/a/61713634 28.05.2024
        start = time.perf_counter()
        solution_set = self.run()
        runtime = timedelta(seconds=time.perf_counter() - start)

        print_solution(runtime, solution_set)


if __name__ == "__main__":
    sol = np.array(
        [
            [["5vs1", np.nan], ["3vs6", "2vs4"], [np.nan, np.nan]],
            [["6vs2", np.nan], ["1vs4", "3vs5"], [np.nan, np.nan]],
            [["5vs4", np.nan], ["1vs6", np.nan], ["3vs2", np.nan]],
            [["6vs4", np.nan], ["1vs3", np.nan], ["5vs2", np.nan]],
            [["1vs2", np.nan], ["3vs4", "5vs6"], [np.nan, np.nan]],
            [["6vs3", np.nan], ["4vs1", np.nan], ["2vs5", np.nan]],
            [["3vs1", np.nan], ["4vs2", np.nan], ["6vs5", np.nan]],
            [["1vs5", np.nan], ["4vs3", "2vs6"], [np.nan, np.nan]],
            [["4vs5", np.nan], ["2vs3", np.nan], ["6vs1", np.nan]],
            [["4vs6", np.nan], ["5vs3", "2vs1"], [np.nan, np.nan]],
        ]
    )

    algo_config = {
        "n": 6,
        "t": 2 / 3,
        "s": 2,
        "r": 3,
        "p": np.array([
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
        ]),
    }

    lns = ALNS(algo_config=algo_config, time_out=5, start_solution=sol)
    lns.run()
