"""This file contains different neighborhoods for the Metaheuristics"""

# Third party libraries
import numpy as np


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
