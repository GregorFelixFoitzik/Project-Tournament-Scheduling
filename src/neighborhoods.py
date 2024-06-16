"""This file contains different neighborhoods for the Metaheuristics"""

# Standard library
import itertools

# Third party libraries
import numpy as np

# Project specific library
from validation import validate
from helper import (
    generate_possible_game_combinations_per_week,
    generate_possible_weekly_combinations,
    get_profits_per_week,
    compute_profit,
)


def select_random_weeks(
    sol: np.ndarray, number_of_weeks: int
) -> tuple[np.ndarray, np.ndarray]:
    # Destroy 2 weeks randomly
    weeks_changed = np.random.choice(
        a=list(range(sol.shape[0])), size=number_of_weeks, replace=False
    )
    games = sol[weeks_changed].copy()

    return weeks_changed, games


def insert_games_random_week(
    sol: np.ndarray, games_week: np.ndarray, week_changed: int, number_of_teams: int
) -> np.ndarray:
    games_unique = np.unique(ar=games_week, axis=1)
    games_unique = games_unique[np.logical_not(np.isnan(games_unique))]
    games_unique = games_unique.reshape(int(games_unique.shape[0] / 2), 2)

    sol_without_week = sol.copy()
    sol_without_week[week_changed] = np.full(shape=games_week.shape, fill_value=np.nan)

    # Which teams have to play on monday and how does the new week look like?
    teams_play_on_monday = np.unique(sol_without_week[:, 0])[:-1]
    week_new = np.full(shape=games_week.shape, fill_value=np.nan)

    # Set the monday game
    if (
        np.setdiff1d(ar1=range(1, number_of_teams + 1), ar2=teams_play_on_monday).size
        == 0
    ):
        monday_game_idx = np.random.choice(games_unique.shape[0])
        week_new[0][0] = games_unique[monday_game_idx]
    else:
        monday_game_idx = np.where(
            games_unique
            == np.setdiff1d(ar1=range(1, number_of_teams + 1), ar2=teams_play_on_monday)
        )[0]
        week_new[0][0] = games_unique[monday_game_idx]

    # Randomly distribute the remaiing games
    remaining_games = games_unique[games_unique != games_unique[monday_game_idx]]
    remaining_games = remaining_games.reshape(int(remaining_games.shape[0] / 2), 2)

    random_choice = np.random.choice([1, 2], size=remaining_games.shape[0])

    for game_idx, day_choice in enumerate(iterable=random_choice):
        week_new[day_choice][np.where(np.isnan(week_new[day_choice]))[0][0]] = (
            remaining_games[game_idx]
        )

    return week_new


def select_n_worst_weeks(
    sol: np.ndarray, n: int, profits: np.ndarray, weeks_between: int
) -> tuple[np.ndarray, np.ndarray]:
    week_profits = get_profits_per_week(
        sol=sol, profit=profits, weeks_between=weeks_between
    )
    worst_weeks = np.argsort(week_profits)[:n]
    games = sol[worst_weeks].copy()

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
) -> np.ndarray:
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
            week_new[0][0] = week[0]
            for game in week[1:]:
                games_position = np.argmax(profits[1:, game[0] - 1, game[1] - 1]) + 1
                week_new[games_position][
                    np.where(np.isnan(week_new[games_position]))[0][0]
                ] = game
            sol_new[weeks_changed[i]] = week_new

        # Check if the solution is valid or not
        try:
            validate(sol=sol_new, num_teams=num_teams)
        except AssertionError:
            continue

        profit_new_sol = compute_profit(
            sol=sol_new, profit=profits, weeks_between=weeks_between
        )

        # If the solution is valid: Does the solution give a higher profit?
        if profit_new_sol > max_profit:
            max_profit =profit_new_sol
            max_sol = sol_new.copy()
    return max_sol.copy()
