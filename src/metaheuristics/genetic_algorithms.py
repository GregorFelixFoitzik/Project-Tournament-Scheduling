# Standard library
import time

from typing import Union
from datetime import timedelta

# Third party library
import numpy as np

# Project specific library
from src.helper import generate_solution_round_robin_tournament, print_solution

from src.neighborhoods import (
    insert_games_max_profit_per_week,
    insert_games_random_week,
    select_n_worst_weeks,
    select_random_weeks,
)
from src.validation import validate
from src.helper import compute_profit, print_solution


class GeneticAlgorithm:
    def __init__(
        self,
        algo_config: dict[str, Union[int, float, np.ndarray]],
        time_out: int,
        max_num_generations: int,
        max_population_size: int,
        crossover: str,
        crossover_parameters: dict[str, Union[int, float]],
    ) -> None:
        self.n = algo_config["n"]  # int
        self.t = algo_config["t"]  # float
        self.s = algo_config["s"]  # int
        self.r = algo_config["r"]  # int
        # Has shape (3, n, n), so self.p[0] is the profit for monday
        self.p = np.array(algo_config["p"]).reshape((3, self.n, self.n))  # np.array

        self.time_out = time_out

        self.best_solution = np.array([])

        self.all_teams = range(1, self.n + 1)

        self.max_num_generations = max_num_generations
        self.max_population_size = max_population_size

        self.crossover = crossover
        self.crossover_operators = {"uniform_crossover": self.uniform_crossover}
        self.crossover_parameters = crossover_parameters

    def run(self) -> np.ndarray:
        """
        Execute algorithm

        Args:
           None

        Returns:
            None
        """
        # TODO: Generate solutions here
        population = [generate_solution_round_robin_tournament(num_teams=self.n, t=self.t) for _ in range(self.max_population_size)]

        t0 = time.time()
        num_generations = 0
        avg_runtime = 0
        while (
            num_generations <= self.max_num_generations
            and (time.time() - t0) + avg_runtime < self.time_out
        ):
            fitness = [
                compute_profit(
                    sol=sol, profit=np.array(object=self.p) ** 2, weeks_between=self.r
                )
                for sol in population
            ]
            fitness_perc = [fit / np.sum(a=fitness) for fit in fitness]

            new_population = []
            while len(new_population) < self.max_population_size:
                parent_0, parent_1 = np.random.choice(
                    a=range(self.max_population_size),
                    size=2,
                    replace=False,
                    p=fitness_perc,
                )

                if compute_profit(population[parent_0], self.p, self.r) < compute_profit(population[parent_1], self.p, self.r):
                    parent_0, parent_1 = parent_1, parent_0

                child = self.crossover_operators[self.crossover](
                    population[parent_0], population[parent_1]
                )
                child_mutated = mutation_swap_home_away(child=child)

                new_population.append(child_mutated)

            population = np.array(new_population)

        fitness = [
            compute_profit(
                sol=sol, profit=np.array(object=self.p) ** 2, weeks_between=self.r
            )
            for sol in population
        ]
        best_solution = population[np.argmax(fitness)]

        self.best_solution = best_solution

        return best_solution

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


    def uniform_crossover(self, parent_0: np.ndarray, parent_1: np.ndarray) -> np.ndarray:
        child = parent_0.copy()
        weeks_swap = np.random.choice(range(parent_0.shape[0]), self.crossover_parameters['num_swaps'], replace=False)
        child[weeks_swap] = parent_1[weeks_swap]

        try:
            validate(child, self.n)
            return child
        except Exception:
            games = child[np.logical_not(np.isnan(child))].reshape(
                int(child[np.logical_not(np.isnan(child))].shape[0] / 2), 2
            )
            games_unique = np.unique(games, axis=0)

            return child

    def mutation_swap_home_away(self, hild: np.ndarray) -> np.ndarray:
        pass
