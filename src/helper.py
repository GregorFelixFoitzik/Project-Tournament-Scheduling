"""File contains helper functions to apply the Metaheuristics"""

# Standard library
import re
import ast
import itertools

from math import inf

# Third party library
import numpy as np

from typing import Union
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
            algo_config[key] = np.array(object=ast.literal_eval(node_or_string=value))
        elif key == "t":
            algo_config[key] = float(value.strip())
        elif key == "p":
            algo_config[key] = np.array(object=value.strip().split(" ")).astype(
                dtype=int
            )
        else:
            algo_config[key] = int(value.strip())

    return algo_config


def print_solution(runtime: float, solution: np.ndarray = np.array(object=[])) -> None:
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
        formatted_blocks = [np.array(object=block) for block in formatted_blocks]

        # Concatenate the blocks horizontally
        concatenated_array = np.hstack(tup=formatted_blocks)

        # Print the concatenated array using tabulate
        headers = ["Mon", "Mon", "Mon", "Fri", "Fri", "Fri", "Sat", "Sat", "Sat"]
        print(
            tabulate(tabular_data=concatenated_array, tablefmt="grid", headers=headers)
        )

    else:
        print("The solution is None and cannot be processed.")


def print_feasible_solution(sol: np.ndarray, runtime: float, profit: float) -> None:
    """Print a feasible solution.

    This function is only called if the solution is feasible. The output design is
    specified in the task description.

    Args:
        sol (np.ndarray): The solution array, that should be printed.
        runtime (float): How long dif the Metaheuristic run?
        profit (float): Profit of the solution.
    """
    print("### RESULT: Feasible")
    print(f"### OBJECTIVE: {profit}")
    weekdays = {0: "M", 1: "F", 2: "S"}
    for week_num, week in enumerate(iterable=sol):
        for weekday_num, games in enumerate(iterable=week):
            for game in games:
                if not np.isnan(np.unique(game)).any():
                    print(
                        f"### Game {week_num+1}-{weekdays[weekday_num]}: "
                        f"{game[0]} {game[1]}"
                    )

    print(f"### CPU-TIME : {np.round(a=runtime, decimals=2)}")


def compute_profit(sol: np.ndarray, profit: np.ndarray, weeks_between: int) -> float:
    """Compute the profit for a given solution.

    Call the function that computes the profit for each week.

    Args:
        sol (np.ndarray): The solution array, that should be printed.
        profit (np.ndarray): Array containing the profits for each week and game
        weeks_between (int): How many games should be between two identical games?

    Returns:
        float: The overall profit for the scheduling.
    """
    return np.sum(
        a=get_profits_per_week(sol=sol, profits=profit, weeks_between=weeks_between)
    )


def get_profit_game(
    sol: np.ndarray,
    game: np.ndarray,
    profits: np.ndarray,
    week_num: int,
    weeks_between: int,
) -> float:
    """Get the profit for one game.

    Compute the profit for one game. If necessary, reduce the profit due to a violation
    of the constraint, that two teams should play again after $r$ weeks.

    Args:
        sol (np.ndarray): The solution array, that should be printed.
        game (np.ndarray): Array contains the game.
        profits (np.ndarray): Array contains the profits for a given day.
        week_num (int): Which week are we in?
        weeks_between (int): How many weeks should be between two games?

    Returns:
        float: The computed profit.
    """
    location_game = np.where(np.isin(element=sol, test_elements=game).all(axis=3))[0]
    # If the other games was also destroyed, return the profit becaus we start with
    #   the smallest week
    if location_game.size == 0:
        return profits[int(game[0]) - 1][int(game[1]) - 1]
    game_other_location = np.setdiff1d(ar1=location_game, ar2=week_num)[0]
    # Is the other game later than the current week?
    if game_other_location > week_num:
        # Yes: Add the the normal profit.
        profit_game = profits[int(game[0]) - 1][int(game[1]) - 1]
    else:
        # No: Check if the differenc between the current-week and other
        #   week is in the interval [1, r)
        if 1 <= week_num - game_other_location < weeks_between:
            # Reduce the profit
            profit_game = profits[int(game[0]) - 1][int(game[1]) - 1] / (
                1 + (week_num - game_other_location) ** 2
            )
        else:
            profit_game = profits[int(game[0]) - 1][int(game[1]) - 1]

    return profit_game


def get_profits_of_week(
    sol: np.ndarray, profits: np.ndarray, weeks_between: int, week_num: int
    ) -> float:
    """Compute the profit for the given week.

    Compute the profit for the given week and return it.

    Args:
        sol (np.ndarray): The solution array, that should be printed.
        profits (np.ndarray): Array contains the profits for a given day.
        weeks_between (int): How many weeks should be between two games?
        week: From which week should the profit be calculated?

    Returns:
        float: profit as float for given week.
    """
    week = sol[week_idx]
    for i, games in enumerate(iterable=week):
            # Extract the games of the given day
            games_flatten = games.flatten()
            games_flatten_no_nan = np.logical_not(np.isnan(games.flatten()))
            games = games_flatten[games_flatten_no_nan].reshape(
                int(games_flatten[games_flatten_no_nan].shape[0] / 2), 2
            )

            # If the day is monday and only one game is assigned to the monday slot
            if i == 0 and games.shape[0] == 1:
                game = games[0]
                sum_week += get_profit_game(
                    sol=sol,
                    game=game,
                    profits=profits[0],
                    week_num=week_num,
                    weeks_between=weeks_between,
                )
                continue
            # If the day is monday and multiple games are assigned to this slot
            if i == 0 and np.unique(games, axis=0).shape[1] > 1:
                min_profit = float(inf)

                # Iterate over the different games on monday
                for game in games:
                    profit_game = get_profit_game(
                        sol=sol,
                        game=game,
                        profits=profits[0],
                        week_num=week_num,
                        weeks_between=weeks_between,
                    )

                    if profit_game < min_profit:
                        min_profit = profit_game
                sum_week += min_profit
                continue

            # If the day is not monday: iterate over the given games
            for game in games:
                sum_week += get_profit_game(
                    sol=sol,
                    game=game,
                    profits=profits[i],
                    week_num=week_num,
                    weeks_between=weeks_between,
                )
    return sum_week


def get_profits_per_week(
    sol: np.ndarray, profits: np.ndarray, weeks_between: int
) -> list[float]:
    """Compute the profits for each week.

    Iterate over the different weeks and compute the profit for each week and append
    the value to a list, that is returned.

    Args:
        sol (np.ndarray): The solution array, that should be printed.
        profits (np.ndarray): Array contains the profits for a given day.
        weeks_between (int): How many weeks should be between two games?

    Returns:
        list[float]: List contains the profits as float for each week.
    """
    week_profits = []

    # Extract all games, that play during that week
    # games_all_flatten = sol.flatten()
    # games_all_flatten_no_nan = np.logical_not(np.isnan(sol.flatten()))
    # games_all = games_all_flatten[games_all_flatten_no_nan].reshape(
    # int(games_all_flatten[games_all_flatten_no_nan].shape[0] / 2), 2
    # )

    # Iterate over each week
    for week_num, week in enumerate(iterable=sol):
        sum_week = 0
        # Iterate over the different days
        for i, games in enumerate(iterable=week):
            # Extract the games of the given day
            games_flatten = games.flatten()
            games_flatten_no_nan = np.logical_not(np.isnan(games.flatten()))
            games = games_flatten[games_flatten_no_nan].reshape(
                int(games_flatten[games_flatten_no_nan].shape[0] / 2), 2
            )

            # If the day is monday and only one game is assigned to the monday slot
            if i == 0 and games.shape[0] == 1:
                game = games[0]
                sum_week += get_profit_game(
                    sol=sol,
                    game=game,
                    profits=profits[0],
                    week_num=week_num,
                    weeks_between=weeks_between,
                )
                continue
            # If the day is monday and multiple games are assigned to this slot
            if i == 0 and np.unique(games, axis=0).shape[1] > 1:
                min_profit = float(inf)

                # Iterate over the different games on monday
                for game in games:
                    profit_game = get_profit_game(
                        sol=sol,
                        game=game,
                        profits=profits[0],
                        week_num=week_num,
                        weeks_between=weeks_between,
                    )

                    if profit_game < min_profit:
                        min_profit = profit_game
                sum_week += min_profit
                continue

            # If the day is not monday: iterate over the given games
            for game in games:
                sum_week += get_profit_game(
                    sol=sol,
                    game=game,
                    profits=profits[i],
                    week_num=week_num,
                    weeks_between=weeks_between,
                )

        week_profits.append(sum_week)

    # print(time.time() - t0)

    return week_profits


def generate_possible_game_combinations_per_week(
    games_encoded: list[int],
    num_repetitions: int,
    games: np.ndarray,
    all_teams: list[int],
) -> list[int]:
    """Generate possible game-combinations for each week.

    This function is only used if there are not that many teams.

    Args:
        games_encoded (list[int]): List contains the encoding of the games.
        num_repetitions (int): How long should be a combinations?
        games (np.ndarray): All games for the tournament.
        all_teams (list[int]): List containing each team.

    Returns:
        list[int]: List containing the different game indices for each combination.
    """
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
    """Generate possible weekly combinations.

    This function considers all hard side-constraints.

    Args:
        possible_combinations_tmp_idx (list[int]): List containing the different game
            indices for each combination.
        weeks_changed (np.ndarray): Which weeks were changed due to the Metaheuristic.
        games (np.ndarray): All games that play during that week.

    Returns:
        list[np.ndarray]: The possible weekly-combinations.
    """
    possible_weekly_combinations = []
    # Create all possible weekly combinations
    weekly_combinations = np.array(
        object=list(
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
def generate_solution_round_robin_tournament(
    num_teams: int, t: float, random_team_order: bool
) -> np.ndarray:
    """Cretea a feasible initial solution via a round-robin tournament.

    Dependingj on the input parameters, the team order is shuffeled or not.

    Args:
        num_teams (int): How many teams take part in this tournament.
        t (float): Fraction to compute the number of games on friday and saturday.
        random_team_order (bool): Whether the team order should be random or not for 
            the start of the rr-tournament.

    Returns:
        np.ndarray: A possible solution that is feasible as ndarray.
    """
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

    # Create the teams array for the rr-tournament: for example [[1,2,3], [4,5,6]]
    teams = list(range(1, num_teams + 1))
    if random_team_order:
        teams = np.random.choice(
            a=teams, size=np.array(object=teams).shape, replace=False
        ).tolist()
    teams = np.array(object=teams).reshape(2, int(num_teams / 2)).tolist()

    # Create the first half of the campaign via applying the rr-tournament-algorithm
    for week in range(num_teams - 1):
        solution_shortened[week] = [
            np.array(object=teams)[:, i].tolist() for i in range(int(num_teams / 2))
        ]
        upper_row = [teams[0][0], teams[1][0]] + teams[0][1:-1]
        lower_row = teams[1][1:] + [teams[0][-1]]

        teams = [upper_row, lower_row]

    # Add the second half to the solution
    solution_extended = np.array(object=solution_shortened + solution_shortened)

    # Invert the second half of the solution
    for i in range(num_teams - 1, solution_extended.shape[0]):
        week = solution_extended[i]
        week[:, [0, 1]] = week[:, [1, 0]]

    # Get how many games should be played on monday
    num_games_monday = max(
        1, solution_extended.shape[1] - 2 * int(np.ceil(solution_extended.shape[1] * t))
    )

    # Iterate over the weeks and assign the games to the different days for each week
    for i, week in enumerate(iterable=solution_extended):
        # The first num_games_monday are monday games
        solution[i][0][:num_games_monday] = week[:num_games_monday]

        # Assign all other games to friday
        num_games_friday = int(np.ceil(solution_extended.shape[1] * t))
        solution[i][1][: num_games_monday + num_games_friday - num_games_monday] = week[
            num_games_monday : num_games_monday + num_games_friday
        ]

        # If there are remaining games: Assign them to saturday
        if week[num_games_monday + num_games_friday :].shape[0] > 0:
            remaining_games = week[num_games_monday + num_games_friday :]
            solution[i][2][: remaining_games.shape[0]] = remaining_games

    validate(sol=solution, num_teams=num_teams)

    return solution
