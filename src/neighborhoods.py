"""This file contains different neighborhoods for the Metaheuristics"""

# Standard library

# Third party libraries
import numpy as np

# Project specific library
from src.validation import validate
from src.helper import (
    generate_possible_game_combinations_per_week,
    generate_possible_weekly_combinations,
    get_profit_game,
    get_profits_per_week,
    compute_profit,
)


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

    if weeks_changed.size == 0:
        print('select_random_weeks')

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
        week_new[0][: pos_games] = games_unique[monday_games_idx[:pos_games]]

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

    if a.shape[0] != int(number_of_teams / 2):
        print("insert_games_random_week")

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

    if worst_weeks.size == 0:
        print('select_n_worst_weeks')

    return worst_weeks, games


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
        sol[current_week][0][: pos_games] = games_unique[
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
        else:
            print("BIIIIIIG problem")
            raise Exception

    return sol[current_week]


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
