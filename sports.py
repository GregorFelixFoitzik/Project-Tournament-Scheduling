# Standard library
import os

# Source: https://stackoverflow.com/a/48665619, accessed 25th June
# Might need adaption if other CPU is used
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import re
import ast
import time
import argparse
import itertools

from math import inf
from typing import Union

# Third party libraries
import numpy as np

# Copied from https://docs.python.org/3/library/argparse.html, accessed 28.05.2024
parser = argparse.ArgumentParser(description="Input for algorithms")
# ChatGPT fixed the issues that default values were not used. => python main.py works now
parser.add_argument(
    "timeout", type=float, default=30, help="Timeout duration in seconds"
)
parser.add_argument(
    "path_to_instance",
    type=str,
    default="data/example.in",
    help="Path to the input file",
)


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
        self.rc = rc

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


def select_random_weeks(
    sol: np.ndarray, number_of_weeks: int, tabu_list: list[int] = []
) -> tuple[np.ndarray, np.ndarray]:
    """Select random weeks from the solution.

    If a tabu-list is present, only use the weeks which are not in the tabu-list.

    Args:
        sol (np.ndarray): The complete solution, that should be used.
        number_of_weeks (int): How many weeks should be selected?
        tabu_list (list[int], optional): The tabu list, that can be empty. Defaults
            to [].

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing the weeks that have been
            selected and the corresponding games.
    """
    # Destroy n weeks randomly
    if tabu_list:
        possible_weeks = np.setdiff1d(ar1=list(range(sol.shape[0])), ar2=tabu_list)
        number_of_weeks = min(number_of_weeks, possible_weeks.shape[0])
        weeks_changed = np.random.choice(
            a=possible_weeks,
            size=number_of_weeks,
            replace=False,
        )
    else:
        weeks_changed = np.random.choice(
            a=list(range(sol.shape[0])), size=number_of_weeks, replace=False
        )
    games = sol[weeks_changed].copy()

    return weeks_changed, games


def insert_games_random_week(
    sol: np.ndarray,
    games_week: np.ndarray,
    week_changed: int,
    number_of_teams: int,
    t: float,
) -> np.ndarray:
    """Insert games randomly in a destroyed week.

    The constraint that every team plays on monday is still considered.

    Args:
        sol (np.ndarray): Solution, which should be updated.
        games_week (np.ndarray): The different games that should be in this week.
        week_changed (int): Number of the week that is changed
        number_of_teams (int): How many teams take part in this tournament.
        t (float): Fraction that sets the number of games on firday and saturday.

    Returns:
        np.ndarray: Thw week with a random assignment.
    """
    # Get unique games for a given week
    games_week_reshape = games_week.reshape(
        (games_week.shape[0] * games_week.shape[1], 2)
    )
    games_week_idx = np.logical_not(np.isnan(games_week_reshape)).all(axis=1)
    games_unique = games_week_reshape[games_week_idx]

    sol_without_week = sol.copy()
    sol_without_week[week_changed] = np.full(shape=games_week.shape, fill_value=np.nan)

    num_games_monday = max(
        1, int(number_of_teams / 2) - 2 * np.ceil(number_of_teams / 2 * t)
    )
    num_games_fri_sat = np.ceil(number_of_teams / 2 * t)

    # Which teams have to play on monday and how does the new week look like?
    teams_play_on_monday = np.unique(sol_without_week[:, 0])[:-1]
    week_new = np.full(shape=games_week.shape, fill_value=np.nan)
    # Set the monday game
    games_added = []
    start_index_monday = 0
    # If team(s) do not play on monday, add them first
    if (
        np.setdiff1d(ar1=range(1, number_of_teams + 1), ar2=teams_play_on_monday).size
        != 0
    ):
        monday_games_idx = np.where(
            np.isin(
                games_unique,
                np.setdiff1d(
                    ar1=range(1, number_of_teams + 1), ar2=teams_play_on_monday
                ),
            ).any(axis=1)
        )[0]
        pos_games = min(sol.shape[2], monday_games_idx.shape[0])
        week_new[0][:pos_games] = games_unique[monday_games_idx[:pos_games]]

        num_games_monday -= pos_games
        start_index_monday += pos_games

        games_added += monday_games_idx[:pos_games].tolist()

    # If there are remaining games that should be added to the monday slot
    if num_games_monday > 0:
        monday_games_idx = np.random.choice(
            a=list(range(games_unique.shape[0])), size=num_games_monday, replace=False
        )
        week_new[0][start_index_monday:num_games_monday] = games_unique[
            monday_games_idx
        ]
        games_added += monday_games_idx.tolist()

    # Randomly distribute the remaiing games
    remaining_games_idx = np.setdiff1d(list(range(games_unique.shape[0])), games_added)
    days_per_game = np.random.choice(
        a=[0 for _ in range(int(num_games_fri_sat))]
        + [1 for _ in range(int(num_games_fri_sat))],
        size=len(remaining_games_idx),
        replace=False,
    )
    # Add the games to the new week
    games_per_day = [0, 0]
    for i, day in enumerate(days_per_game):
        week_new[day + 1][games_per_day[day]] = games_unique[remaining_games_idx[i]]
        games_per_day[day] += 1

    a = week_new[np.logical_not(np.isnan(week_new))].reshape(
        int(week_new[np.logical_not(np.isnan(week_new))].shape[0] / 2), 2
    )

    return week_new


def select_n_worst_weeks(
    sol: np.ndarray,
    n: int,
    profits: np.ndarray,
    weeks_between: int,
    tabu_list: list = [],
) -> tuple[np.ndarray, np.ndarray]:
    """Select the $n$ worst weeks from the solution.

    Args:
        sol (np.ndarray): Solution as ndarray, that should be improved.
        n (int): How many teams should be selected.
        profits (np.ndarray): Profits for each day and game.
        weeks_between (int): How many weeks should be between a game of two teams?
            This parameter is used to compute the profits.
        tabu_list (list, optional): Tabu-list for weeks that should not be used.
            Defaults to [].

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing the games and the week-numbers
            of the worst $n$ weeks.
    """
    week_profits = get_profits_per_week(
        sol=sol, profits=profits, weeks_between=weeks_between
    )
    if tabu_list:
        worst_weeks = np.setdiff1d(np.argsort(week_profits), tabu_list)[:n]
    else:
        worst_weeks = np.argsort(week_profits)[:n]

    games = sol[worst_weeks].copy()

    return worst_weeks, games


def reorder_week_max_profit(
    sol: np.ndarray,
    profits: np.ndarray,
    games: np.ndarray,
    num_teams: int,
    t: float,
    current_week: int,
    weeks_between: int,
) -> np.ndarray:
    """Reorder the given week, so the profit is maximized.

    Args:
        sol (np.ndarray): Solution which should be improved.
        profits (np.ndarray): Profits for each week and game.
        games (np.ndarray): The games that should be reorganized.
        num_teams (int): How many teams take part in the tournament.
        t (float): Fraction that sets the number of games on friday and saturday.
        current_week (int): The week that should be improved.
        weeks_between (int): Number of weeks between a game of two teams.

    Returns:
        np.ndarray: The updated week.
    """
    # Get all games for the given week
    games_flatten = games.flatten()
    games_flatten_no_nan = np.logical_not(np.isnan(games.flatten()))
    games_unique = games_flatten[games_flatten_no_nan].reshape(
        int(games_flatten[games_flatten_no_nan].shape[0] / 2), 2
    )

    # For each of the games compute the profits for each of the days
    profits_per_game = np.full(shape=(games_unique.shape[0], 3), fill_value=np.nan)
    for i, game in enumerate(iterable=games_unique):
        profit_monday = get_profit_game(
            sol=sol,
            game=game,
            profits=profits[0],
            week_num=current_week,
            weeks_between=weeks_between,
        )
        profit_friday = get_profit_game(
            sol=sol,
            game=game,
            profits=profits[1],
            week_num=current_week,
            weeks_between=weeks_between,
        )
        profit_saturday = get_profit_game(
            sol=sol,
            game=game,
            profits=profits[2],
            week_num=current_week,
            weeks_between=weeks_between,
        )

        profits_per_game[i] = [profit_monday, profit_friday, profit_saturday]

    # Check which games should be played on monday
    start_index_monday = 0
    num_teams_monday = max(1, int(num_teams / 2) - 2 * np.ceil(num_teams / 2 * t))
    num_games_fri_sat = np.ceil(num_teams / 2 * t)
    teams_on_monday = np.setdiff1d(
        ar1=list(range(1, num_teams + 1)), ar2=np.unique(ar=sol[:, 0])[:-1]
    )
    games_added = []
    if teams_on_monday.size > 0:
        # Insert the games with the teams that do not play on monday
        games_forced_monday_idx = np.where(
            np.isin(games_unique, teams_on_monday).any(axis=1)
        )[0]
        pos_games = min(sol.shape[2], games_forced_monday_idx.shape[0])
        sol[current_week][0][:pos_games] = games_unique[
            games_forced_monday_idx[:pos_games]
        ]
        start_index_monday += pos_games
        num_teams_monday -= pos_games
        games_added += games_forced_monday_idx[:pos_games].tolist()

    # If the monday slot is not full: Add more games but with the highest profit
    if num_teams_monday > 0:
        games_on_monday = np.argsort(profits_per_game[:, 0])[::-1][:num_teams_monday]
        games_added += games_on_monday.tolist()

        sol[current_week][0][start_index_monday:num_teams_monday] = games_unique[
            games_on_monday
        ]

    # Iterate over the profits for firday and saturday in descending order and add the
    #   games with the biggest profit first
    profits_friday_saturday = np.sort(profits_per_game[:, 1:].reshape(1, -1)[0])[::-1]
    num_games_per_fri_sat = [0, 0]
    for profit in profits_friday_saturday:
        # Get the game that corresponds to this profit
        possible_game = np.setdiff1d(
            ar1=np.where(
                np.isin(element=profits_per_game[:, 1:], test_elements=[profit])
            )[0],
            ar2=games_added,
        )
        if possible_game.size == 0:
            continue

        # Check on which day the game should be play (friday or saturday)
        possible_game = possible_game[0]
        day_max_profit = np.where(profits_per_game[possible_game][1:] == profit)[0][0]
        games_added.append(possible_game)
        if day_max_profit == 0 and num_games_per_fri_sat[0] < num_games_fri_sat:
            sol[current_week][1][num_games_per_fri_sat[0]] = games_unique[
                possible_game
            ]  #
            num_games_per_fri_sat[0] += 1
        elif day_max_profit == 0 and num_games_per_fri_sat[1] < num_games_fri_sat:
            sol[current_week][2][num_games_per_fri_sat[1]] = games_unique[
                possible_game
            ]  #
            num_games_per_fri_sat[1] += 1
        elif day_max_profit == 1 and num_games_per_fri_sat[1] < num_games_fri_sat:
            sol[current_week][2][num_games_per_fri_sat[1]] = games_unique[
                possible_game
            ]  #
            num_games_per_fri_sat[1] += 1
        elif day_max_profit == 1 and num_games_per_fri_sat[0] < num_games_fri_sat:
            sol[current_week][1][num_games_per_fri_sat[0]] = games_unique[
                possible_game
            ]  #
            num_games_per_fri_sat[0] += 1

    return sol[current_week]


def insert_games_max_profit_per_week(
    sol: np.ndarray,
    games_old: np.ndarray,
    games_encoded: list[int],
    num_repetitions: int,
    games: np.ndarray,
    all_teams: list[int],
    weeks_changed: np.ndarray,
    profits: np.ndarray,
    num_teams: int,
    weeks_between: int,
    t: float,
) -> np.ndarray:
    """Insert the games for a week, so the profit is maximized.

    This function creates possible combinations of weeks and checks, which combination
    maximizes the profit. Note that this function is only applied, when not many teams
    take part in the tournament.

    Args:
        sol (np.ndarray): The solution that should be improved
        games_old (np.ndarray): Games that were previously assigned to the changed
            weeks.
        games_encoded (list[int]): Games encoded to numbers for faster computation.
        num_repetitions (int): How often should a week be created.
        games (np.ndarray): The games that should be inserted.
        all_teams (list[int]): List containing all teams.
        weeks_changed (np.ndarray): Which weeks were changed beofre this operation.
        profits (np.ndarray): The profits for each day and game.
        num_teams (int): How many teams take part in this tournament.
        weeks_between (int): Number of weeks between a game of two teams.
        t (float): Fraction that sets the number of games on friday and saturday.

    Returns:
        np.ndarray: A new solution, that maximizes the profit.
    """
    # Get the possible combinations
    possible_combinations_tmp_idx = generate_possible_game_combinations_per_week(
        games_encoded=games_encoded,
        num_repetitions=num_repetitions,
        games=games,
        all_teams=all_teams,
    )

    possible_weekly_combinations = generate_possible_weekly_combinations(
        possible_combinations_tmp_idx=possible_combinations_tmp_idx,
        weeks_changed=weeks_changed,
        games=games,
    )

    # Get max profit when inserted in solution for each combination
    max_profit = 0
    max_sol = sol.copy()
    for weekly_combination in possible_weekly_combinations:
        sol_new = sol.copy()
        # Insert each weekly-combination into the solution
        for i, week in enumerate(iterable=weekly_combination):
            week_new = np.full(shape=games_old[0].shape, fill_value=np.nan)
            num_games_monday = week.shape[0] - int(week.shape[0] * t)
            week_new[0][:num_games_monday] = week[:num_games_monday]
            for game in week[num_games_monday:]:
                games_position = np.argmax(a=profits[1:, game[0] - 1, game[1] - 1]) + 1
                week_new[games_position][
                    np.where(np.isnan(week_new[games_position]))[0][0]
                ] = game
            sol_new[weeks_changed[i]] = week_new

        # Check if the solution is valid or not (if yes, continue)
        try:
            validate(sol=sol_new, num_teams=num_teams)
        except AssertionError:
            continue

        profit_new_sol = compute_profit(
            sol=sol_new, profit=profits, weeks_between=weeks_between
        )

        # If the solution is valid: Does the solution give a higher profit?
        if profit_new_sol > max_profit:
            max_profit = profit_new_sol
            max_sol = sol_new.copy()
    return max_sol.copy()


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


def random_reorder_weeks(
    sol: np.ndarray, games: np.ndarray, weeks_changed: np.ndarray
) -> np.ndarray:
    """Randomly reorder the weeks.

    Args:
        sol (np.ndarray): Solution that should be improved.
        games (np.ndarray): Games for each week
        weeks_changed (np.ndarray): Indices of the weeks that were changed.

    Returns:
        np.ndarray: The updated solution.
    """
    new_order = np.random.choice(
        a=list(range(games.shape[0])), size=games.shape[0], replace=False
    )
    sol[weeks_changed] = games[new_order]
    return sol


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

    return week_profits


def validate(
    sol: np.ndarray,
    num_teams: int,
) -> bool:
    """Validate the given solution by checking the constraints.

    Args:
        sol (np.ndarray): The solution that should be checked.
        num_teams (int): How many teams take part in the tournament?

    Returns:
        bool: Whether the solution is valid or not.
    """
    weeks = 2 * (num_teams - 1)

    feasible_uniqueness = uniqueness(sol=sol, num_teams=num_teams, weeks=weeks)
    feasible_check_games = check_games(sol=sol, num_teams=num_teams)
    feasible_each_team_monday = every_team_on_monday(sol=sol, num_teams=num_teams)
    feasible_every_team_every_week = every_team_every_week(sol=sol, num_teams=num_teams)

    return (
        feasible_uniqueness
        and feasible_check_games
        and feasible_each_team_monday
        and feasible_every_team_every_week
    )


def uniqueness(sol: np.ndarray, num_teams: int, weeks: int) -> bool:
    """Check for uniqueness.

    Each game should be only once in the solution.

    Args:
        sol (np.ndarray): The solution that should be checked.
        num_teams (int): How many teams take part in the tournament.
        weeks (int): How many weeks the tounament is long.

    Returns:
        bool: If there are no duplicate games.
    """
    # Source: https://www.pythonpool.com/flatten-list-python/, accessed 29.05.2023
    games = sol[np.logical_not(np.isnan(sol))].reshape(
        int(sol[np.logical_not(np.isnan(sol))].shape[0] / 2), 2
    )
    games_unique = np.unique(games, axis=0)

    assert games.shape[0] == games_unique.shape[0] and (
        games_unique.shape[0] == ((num_teams / 2) * weeks)
    ), "uniqueness"

    return games.shape[0] == games_unique.shape[0] and (
        games_unique.shape[0] == ((num_teams / 2) * weeks)
    )


def check_games(sol: np.ndarray, num_teams: int) -> bool:
    """Check that every game is in the solution.

    Args:
        sol (np.ndarray): The solution that should be checked.
        num_teams (int): How many teams take part in the tournament.

    Returns:
        bool: If all games are in the solution.
    """
    games_required = np.array(
        [
            [i, j] if i != j else [np.nan, np.nan]
            for j in range(1, num_teams + 1)
            for i in range(1, num_teams + 1)
        ]
    )
    games_required = games_required[np.logical_not(np.isnan(games_required))]
    games_required = games_required.reshape(int(games_required.shape[0] / 2), 2)

    sol = sol[np.logical_not(np.isnan(sol))]
    sol = sol.reshape(int(sol.shape[0] / 2), 2)

    games_in_sol = 0
    for game in games_required:
        if game in sol:
            games_in_sol += 1

    assert games_required.shape[0] == games_in_sol, "XvsY and YvsX"

    return games_required.shape[0] == games_in_sol


def every_team_on_monday(sol: np.ndarray, num_teams: int) -> bool:
    """Check, that every team plays at least once on monday.

    Args:
        sol (np.ndarray): The solution that should be checked.
        num_teams (int): How many teams take part in the tournament.

    Returns:
        bool: If every team plays at least once on monday.
    """
    teams = [team for team in range(1, num_teams + 1)]

    monday_games = sol[:, 0]
    monday_games = monday_games[np.logical_not(np.isnan(monday_games))]
    monday_games = monday_games.reshape(int(monday_games.shape[0] / 2), 2)

    unique_values_monday_games = np.unique(monday_games)

    assert unique_values_monday_games.shape[0] == len(teams) and np.logical_not(
        False in (unique_values_monday_games == teams)
    ), "Every team on monday"

    return np.logical_not(False in (unique_values_monday_games == teams))


def every_team_every_week(sol: np.ndarray, num_teams: int) -> bool:
    """Check that every team plays every week.

    Args:
        sol (np.ndarray): The solution that should be checked.
        num_teams (int): How many teams take part in the tournament.

    Returns:
        bool: If every team plays every week.
    """
    for i, week in enumerate(sol):
        assert np.all(
            a=np.unique(week)[:-1] == list(range(1, num_teams + 1))
        ), f"In week {i} not all teams play!"

    return True


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


def generate_solution_round_robin_tournament(
    num_teams: int, t: float, random_team_order: bool
) -> tuple[np.ndarray, float]:
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

    runtime_construction = time.process_time()  # sum(os.times()[:2])

    validate(sol=solution, num_teams=num_teams)

    return solution, runtime_construction


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


if __name__ == "__main__":
    # Extract command line arguments
    args = parser.parse_args()
    path_to_file = args.path_to_instance
    timeout = args.timeout

    algo_config = read_in_file(path_to_file=path_to_file)

    start_time = time.process_time()  # Start time measurement
    start_sol, rc = generate_solution_round_robin_tournament(
        num_teams=int(algo_config["n"]),
        t=float(algo_config["t"]),
        random_team_order=False,
    )
    metaheuristic = LNS(
        algo_config=algo_config,
        timeout=timeout,
        start_solution=start_sol,
        rc=rc,
    )
    new_sol = metaheuristic.run()
    end_time = time.process_time()  # End time measurement
    duration = end_time - start_time

    profit_start = compute_profit(
        sol=start_sol,
        profit=np.array(object=list(algo_config["p"])).reshape(
            (3, algo_config["n"], algo_config["n"])
        ),
        weeks_between=int(algo_config["r"]),
    )
    profit_meta = compute_profit(
        sol=new_sol, profit=metaheuristic.p, weeks_between=int(algo_config["r"])
    )

    solution_valid = validate(sol=new_sol, num_teams=algo_config["n"])

    if solution_valid:
        print_feasible_solution(sol=new_sol, runtime=duration, profit=profit_meta)
    else:
        print(f"### RESULT: {timeout}")
