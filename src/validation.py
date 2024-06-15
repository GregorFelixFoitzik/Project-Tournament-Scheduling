"""File contains the validation of the results"""

# Standard library
import itertools

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

    return feasible_uniqueness and feasible_check_games and feasible_each_team_monday and feasible_every_team_every_week


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
    games_required = games_required.reshape(int(games_required.shape[0]/2), 2)

    sol = sol[np.logical_not(np.isnan(sol))]
    sol = sol.reshape(int(sol.shape[0]/2), 2)

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
    monday_games = monday_games.reshape(int(monday_games.shape[0]/2), 2)

    unique_values_monday_games = np.unique(monday_games)
    # print(
        # "Every team plays at least once "
        # f"on monday: {np.logical_not(False in (unique_values_monday_games == teams))}"
    # )

    assert unique_values_monday_games.shape[0] == len(teams) and np.logical_not(False in (unique_values_monday_games == teams)), "Every team on monday"

    return np.logical_not(False in (unique_values_monday_games == teams))

def every_team_every_week(sol: np.ndarray, num_teams: int) -> bool:
    for i, week in enumerate(sol):
        assert np.all(np.unique(week)[:-1] == list(range(1, num_teams + 1))), f"In week {i} not all teams play!"

    return True


# if __name__ == "__main__":
#     num_teams = 6
#     weeks_between = 1
#     profits = [
#         -1.0,
#         15.0,
#         20.0,
#         16.0,
#         28.0,
#         9.0,
#         13.0,
#         -1.0,
#         8.0,
#         12.0,
#         10.0,
#         12.0,
#         19.0,
#         10.0,
#         -1.0,
#         14.0,
#         15.0,
#         13.0,
#         18.0,
#         14.0,
#         12.0,
#         -1.0,
#         13.0,
#         16.0,
#         25.0,
#         15.0,
#         13.0,
#         17.0,
#         -1.0,
#         11.0,
#         10.0,
#         14.0,
#         23.0,
#         21.0,
#         10.0,
#         -1.0,
#         -1.0,
#         8.0,
#         10.0,
#         9.0,
#         5.0,
#         9.0,
#         6.0,
#         -1.0,
#         5.0,
#         11.0,
#         7.0,
#         4.0,
#         8.0,
#         4.0,
#         -1.0,
#         12.0,
#         8.0,
#         6.0,
#         10.0,
#         7.0,
#         4.0,
#         -1.0,
#         4.0,
#         2.0,
#         10.0,
#         7.0,
#         5.0,
#         3.0,
#         -1.0,
#         10.0,
#         4.0,
#         8.0,
#         2.0,
#         7.0,
#         1.0,
#         -1.0,
#         -1.0,
#         2.0,
#         3.0,
#         5.0,
#         3.0,
#         5.0,
#         1.0,
#         -1.0,
#         3.0,
#         2.0,
#         9.0,
#         10.0,
#         7.0,
#         8.0,
#         -1.0,
#         4.0,
#         5.0,
#         2.0,
#         6.0,
#         7.0,
#         5.0,
#         -1.0,
#         2.0,
#         1.0,
#         6.0,
#         8.0,
#         5.0,
#         4.0,
#         -1.0,
#         7.0,
#         8.0,
#         3.0,
#         2.0,
#         7.0,
#         4.0,
#         -1.0,
#     ]

#     t = 2 / 3
#     s = 2

#     # Monday, Friday, Saturday
#     sol = np.array(
#         [
#             [
#                 [[5, 1], [np.nan, np.nan]],
#                 [[3, 6], [2, 4]],
#                 [[np.nan, np.nan], [np.nan, np.nan]],
#             ],
#             [
#                 [[6, 2], [np.nan, np.nan]],
#                 [[1, 4], [3, 5]],
#                 [[np.nan, np.nan], [np.nan, np.nan]],
#             ],
#             [
#                 [[5, 4], [np.nan, np.nan]],
#                 [[1, 6], [np.nan, np.nan]],
#                 [[3, 2], [np.nan, np.nan]],
#             ],
#             [
#                 [[6, 4], [np.nan, np.nan]],
#                 [[1, 3], [np.nan, np.nan]],
#                 [[5, 2], [np.nan, np.nan]],
#             ],
#             [
#                 [[1, 2], [np.nan, np.nan]],
#                 [[3, 4], [5, 6]],
#                 [[np.nan, np.nan], [np.nan, np.nan]],
#             ],
#             [
#                 [[6, 3], [np.nan, np.nan]],
#                 [[4, 1], [np.nan, np.nan]],
#                 [[2, 5], [np.nan, np.nan]],
#             ],
#             [
#                 [[3, 1], [np.nan, np.nan]],
#                 [[4, 2], [np.nan, np.nan]],
#                 [[6, 5], [np.nan, np.nan]],
#             ],
#             [
#                 [[1, 5], [np.nan, np.nan]],
#                 [[4, 3], [2, 6]],
#                 [[np.nan, np.nan], [np.nan, np.nan]],
#             ],
#             [
#                 [[4, 5], [np.nan, np.nan]],
#                 [[2, 3], [np.nan, np.nan]],
#                 [[6, 1], [np.nan, np.nan]],
#             ],
#             [
#                 [[4, 6], [np.nan, np.nan]],
#                 [[5, 3], [2, 1]],
#                 [[np.nan, np.nan], [np.nan, np.nan]],
#             ],
#         ]
#     )


#     validate(
#         sol=sol,
#         num_teams=num_teams,
#         weeks_between=weeks_between,
#         profits=profits,
#         t=t,
#         s=s,
#     )
