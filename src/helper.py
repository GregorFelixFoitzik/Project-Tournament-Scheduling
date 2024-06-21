"""File contains helper functions which might be useful"""

# Standard library
import re
import ast
import itertools

from math import inf

# Third party library
import numpy as np
from typing import Union
from itertools import permutations
from tabulate import tabulate


def read_in_file(path_to_file: str) -> dict[str, Union[float, int, list[int]]]:
    """Read the .in file which contains the configuration.

    After extracting the data, the strings are converted into int, floats and ints.

    Args:
        path_to_file (str): Where is the .in file stored?

    Returns:
        dict[str, Union[float, int, list[int]]]: .in file as dictionary.
    """
    with open(file=path_to_file, mode="r") as file:
        lines = file.readlines()

    algo_config = {}
    # String operations on each line
    for line in lines:
        algo_config[line.split(sep=":")[0]] = line.split(sep=":")[
            1
        ]  # .replace(r'\n', '').strip()

    # Converting the values of the .in file into int, float and list
    for key, value in algo_config.items():
        if re.search(pattern=r"\d*\/\d*", string=value):
            algo_config[key] = int(value.split("/")[0]) / int(value.split("/")[1])
        elif re.search(pattern=r"\[.*\]", string=value):
            # https://stackoverflow.com/a/1894296, accessed 28.05.2024
            algo_config[key] = np.array(ast.literal_eval(node_or_string=value))
        elif key == "t":
            algo_config[key] = float(value.strip())
        elif key == "p":
            algo_config[key] = np.array(value.strip().split(" ")).astype(int)
        else:
            algo_config[key] = int(value.strip())

    return algo_config


def print_solution(runtime: float, solution: np.array = None) -> None:
    """
    Prints the solution as wanted.

    Args:
        runtime (float): CPU-Runtime (in seconds)
        solution (list): contains the current best solution
            index of list:
            #     idx // (self.n/2) => week
            #     idx %  3 => M/F/S (0/1/2)
            value (list): [home, away, profit]
    """
    if solution is not None:
        formatted_blocks = []
        for block in solution:
            formatted_block = [
                [" - ".join(map(str, pair)).replace("0", "") for pair in row]
                for row in block
            ]
            formatted_blocks.append(formatted_block)

        # Convert the formatted blocks into numpy arrays
        formatted_blocks = [np.array(block) for block in formatted_blocks]

        # Concatenate the blocks horizontally
        concatenated_array = np.hstack(formatted_blocks)

        # Print the concatenated array using tabulate
        headers = ["Mon", "Mon", "Mon", "Fri", "Fri", "Fri", "Sat", "Sat", "Sat"]
        print(tabulate(concatenated_array, tablefmt="grid", headers=headers))

    else:
        print("The solution is None and cannot be processed.")


def compute_profit(sol: np.ndarray, profit: np.ndarray, weeks_between: int) -> float:
    return np.sum(get_profits_per_week(sol, profit, weeks_between))


def get_profits_per_week(sol: np.ndarray, profit: np.ndarray, weeks_between: int):
    week_profits = []

    for week_num, week in enumerate(sol):
        sum_week = 0
        for i, games in enumerate(week):
            if i == 0 and np.unique(games, axis=0).shape[0] == 2:
                game = games[np.logical_not(np.isnan(games))]
                sum_week += compute_profit_game(
                    sol,
                    game,
                    profit[0][int(game[0]) - 1][int(game[1]) - 1],
                    weeks_between,
                    week_num,
                )
                continue
            if i == 0 and np.unique(games, axis=0).shape[1] > 2:
                games = games[np.logical_not(np.isnan(games))]
                min_profit = float(inf)

                for game in games:
                    if profit[0][int(game[0]) - 1][int(game[1]) - 1] < min_profit:
                        min_profit = compute_profit_game(
                            sol,
                            game,
                            profit[0][int(game[0]) - 1][int(game[1]) - 1],
                            weeks_between,
                            week_num,
                        )
                sum_week += min_profit
                continue

            for game in games:
                if False in np.isnan(game):
                    sum_week += compute_profit_game(
                        sol,
                        game,
                        profit[i][int(game[0]) - 1][int(game[1]) - 1],
                        weeks_between,
                        week_num,
                    )
        week_profits.append(sum_week)

    return week_profits


def compute_profit_game(
    sol: np.ndarray,
    game: np.ndarray,
    profit_game: float,
    weeks_between: int,
    current_week: int,
) -> float:
    if sol[:current_week].size == 0:
        return profit_game

    for week, games in enumerate(sol[:current_week]):
        games = games[np.logical_not(np.isnan(games))]
        games = games.reshape(int(games.shape[0] / 2), 2).astype(int)
        # Check if the game is in the current week, if no, no move on
        if np.where((games[:, 0] == game[1]) & (games[:, 1] == game[0]))[0].size == 0:
            continue

        # game is in current week: Does they play earlier than expected
        if 1 <= current_week - week < weeks_between:
            return profit_game / (1 + (current_week - week) ** 2)
        else:
            return profit_game

    # If a new game is added
    return profit_game


def get_profit_games_earlier(profit: float, q: np.ndarray) -> np.ndarray:
    return profit / (1 + q ^ 2)


def generate_possible_game_combinations_per_week(
    games_encoded: list[int],
    num_repetitions: int,
    games: np.ndarray,
    all_teams: list[int],
) -> list[int]:
    # Source: https://stackoverflow.com/a/5898031, accessed 11th June
    combinations = itertools.permutations(iterable=games_encoded, r=num_repetitions)
    possible_combinations_tmp_idx = []
    # Iterate over each game combination and check some of the constraints
    for combination_idx in combinations:
        combination = games[list(combination_idx)]
        # Check if all teams play during that week
        if np.unique(ar=combination).size != len(all_teams):
            continue
        if np.all(np.unique(combination) != all_teams):
            continue

        # possible_combinations_tmp.append(np.array(combination))
        possible_combinations_tmp_idx.append(combination_idx)

    return possible_combinations_tmp_idx


def generate_possible_weekly_combinations(
    possible_combinations_tmp_idx: list[int],
    weeks_changed: np.ndarray,
    games: np.ndarray,
) -> list[np.ndarray]:
    possible_weekly_combinations = []
    # Create all possible weekly combinations
    weekly_combinations = np.array(
        list(
            itertools.permutations(
                iterable=possible_combinations_tmp_idx, r=weeks_changed.size
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
            != np.unique(ar=weekly_combination_games, axis=0).shape
        ):
            continue
        possible_weekly_combinations.append(weekly_combination)

    return possible_weekly_combinations


# Described here: https://en.wikipedia.org/wiki/Round-robin_tournament#Circle_method, 
#   accessed 19th June
def generate_solution_round_robin_tournament(num_teams: int, t: float, random_team_order: bool):
    from src.validation import validate

    # Create an empty array of shape: num-weeks x 3 x max num games per day x 2
    solution = np.full(
        shape=(
            2 * (num_teams - 1),
            3,
            max(
                int(num_teams / 2 - int(np.ceil(num_teams / 2 * t))),
                int(np.ceil(num_teams / 2 * t)),
            ),
            2,
        ),
        fill_value=np.nan,
    )

    solution_shortened = np.full(
        shape=(
            num_teams - 1,
            int(num_teams / 2),
            2,
        ),
        fill_value=np.nan,
    ).tolist()

    teams = list(range(1, num_teams + 1))
    if random_team_order:
        team = np.random.choice(teams, np.array(teams).shape, replace=False).tolist()
        print('asd')
    teams = np.array(teams).reshape(2, int(num_teams / 2)).tolist()

    for week in range(num_teams - 1):
        solution_shortened[week] = [
            np.array(teams)[:, i].tolist() for i in range(int(num_teams / 2))
        ]
        upper_row = [teams[0][0], teams[1][0]] + teams[0][1:-1]
        lower_row = teams[1][1:] + [teams[0][-1]]

        teams = [upper_row, lower_row]

    solution_extended = np.array(solution_shortened + solution_shortened)

    # Invert the second half of the solution
    for i in range(num_teams - 1, solution_extended.shape[0]):
        week = solution_extended[i]
        week[:, [0, 1]] = week[:, [1, 0]]

    num_games_monday = max(
        1, solution_extended.shape[1] - 2 * int(np.ceil(solution_extended.shape[1] * t))
    )

    for i, week in enumerate(iterable=solution_extended):
        solution[i][0][:num_games_monday] = week[:num_games_monday]

        num_games_friday = int(np.ceil(solution_extended.shape[1] * t))
        solution[i][1][: num_games_monday + num_games_friday - num_games_monday] = week[
            num_games_monday : num_games_monday + num_games_friday
        ]

        if week[num_games_monday + num_games_friday :].shape[0] > 0:
            remaining_games = week[
                num_games_monday + num_games_friday:
            ]
            solution[i][2][: remaining_games.shape[0]] = remaining_games

    validate(solution, num_teams)

    return solution
