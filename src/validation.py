"""File contains the validation of the results"""

# Standard library

# Third party libraries
import numpy as np


def validate(
    sol: np.ndarray,
    num_teams: int,
) -> bool:
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
    # Source: https://www.pythonpool.com/flatten-list-python/, accessed 29.05.2023
    games = sol[np.logical_not(np.isnan(sol))].reshape(
        int(sol[np.logical_not(np.isnan(sol))].shape[0] / 2), 2
    )
    games_unique = np.unique(games, axis=0)
    # print(
    # f"No duplicates ({games.shape[0]} and {games_unique.shape[0]}): "
    # f"{games.shape[0] == games_unique.shape[0]}"
    # )

    # print(
    # f"Number of games ({games.shape[0]} and {int((num_teams/2)*weeks)}):"
    # f"{games.shape[0] == ((num_teams/2)*weeks)}"
    # )

    assert games.shape[0] == games_unique.shape[0] and (
        games_unique.shape[0] == ((num_teams / 2) * weeks)
    ), "uniqueness"

    return games.shape[0] == games_unique.shape[0] and (
        games_unique.shape[0] == ((num_teams / 2) * weeks)
    )

    # games = np.unique(ar=list(itertools.chain(*list(itertools.chain(*sol)))))
    # print(
    # f"Uniqueness ({np.shape(a=games)[0] - 1} and {int((num_teams/2)*weeks)}):"
    # f"{np.shape(a=games)[0] - 1 == ((num_teams/2)*weeks)}"
    # )

    # return np.shape(a=games)[0] == num_teams


def check_games(sol: np.ndarray, num_teams: int) -> bool:
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

    # print(f"XvsY and YvsX: {games_required.shape[0] == games_in_sol}")

    assert games_required.shape[0] == games_in_sol, "XvsY and YvsX"

    return games_required.shape[0] == games_in_sol


def every_team_on_monday(sol: np.ndarray, num_teams: int) -> bool:
    teams = [team for team in range(1, num_teams + 1)]

    monday_games = sol[:, 0]
    monday_games = monday_games[np.logical_not(np.isnan(monday_games))]
    monday_games = monday_games.reshape(int(monday_games.shape[0] / 2), 2)

    unique_values_monday_games = np.unique(monday_games)
    # print(
    # "Every team plays at least once "
    # f"on monday: {np.logical_not(False in (unique_values_monday_games == teams))}"
    # )

    assert unique_values_monday_games.shape[0] == len(teams) and np.logical_not(
        False in (unique_values_monday_games == teams)
    ), "Every team on monday"

    return np.logical_not(False in (unique_values_monday_games == teams))


def every_team_every_week(sol: np.ndarray, num_teams: int) -> bool:
    for i, week in enumerate(sol):
        # if not np.all(
        # np.unique(week)[:-1] == list(range(1, num_teams + 1))
        # ):
        # return False
        assert np.all(
            np.unique(week)[:-1] == list(range(1, num_teams + 1))
        ), f"In week {i} not all teams play!"

    return True