"""File contains the validation of the results"""

# Standard library
import itertools

# Third party libraries
import numpy as np


def validate(
    sol: list[list[list[str]]],
    num_teams: int,
    weeks_between: int,
    profits: list[float],
    t: float,
    s: int,
) -> bool:
    weeks = 2 * (num_teams - 1)

    feasible_uniqueness = uniqueness(sol=sol, num_teams=num_teams, weeks=weeks)
    feasible_check_games = check_games(sol=sol, num_teams=num_teams)
    feasible_each_team_monday = each_team_on_monday(sol=sol, num_teams=num_teams)

    return feasible_uniqueness and feasible_check_games and feasible_each_team_monday


def uniqueness(sol: list[list[list[str]]], num_teams: int, weeks: int) -> bool:
    # Source: https://www.pythonpool.com/flatten-list-python/, accessed 29.05.2023
    games = np.array(list(itertools.chain(*list(itertools.chain(*sol)))))
    games_unique = np.unique(games, axis=0)
    games_unique = games_unique[np.sum(np.isnan(games_unique), axis=1) == 0]
    games = games[np.sum(np.isnan(games), axis=1) == 0]
    print(
        f"No duplicates ({games.shape[0]} and {games_unique.shape[0]}): "
        f"{games.shape[0] == games_unique.shape[0]}"
    )

    # Source: https://stackoverflow.com/a/22225030, accessed 29.05.2023
    games = games[np.invert(np.unique(np.isnan(games), axis=1).reshape(1, -1)[0])]
    print(
        f"Number of games ({games.shape[0]} and {int((num_teams/2)*weeks)}):"
        f"{games.shape[0] == ((num_teams/2)*weeks)}"
    )

    return games.shape[0] == games_unique.shape[0] and (
        games.shape[0] == num_teams
    )


def check_games(sol: list[list[list[str]]], num_teams: int) -> bool:
    games_required = np.array(
        object=list(
                [
                    {i, j} if i != j else [np.nan, np.nan]
                    for j in range(1,num_teams+1)
                    for i in range(1, num_teams + 1)
                ]
        )
    )
    games_required = games_required[np.sum(np.isnan(games_required), axis=1) == 0]
    
    # Source: https://www.pythonpool.com/flatten-list-python/, accessed 29.05.2023
    sol_flatten = np.array(object=list(itertools.chain(*list(itertools.chain(*sol)))))

    games_in_sol = 0
    for game in games_required:
        if game in sol_flatten:
            games_in_sol += 1

    print(f"XvsY and YvsX: {games_required.shape[0] == games_in_sol}")

    return games_required.shape[0] == games_in_sol


def each_team_on_monday(sol: list[list[list[str]]], num_teams: int) -> bool:
    teams = [team for team in range(1, num_teams + 1)]
    teams_on_monday = []
    for games_monday in sol:
        for game in games_monday[0]:
            if game != [(np.nan, np.nan)]:
                teams_on_monday += game

    print(
        "Each team plays at least once "
        f"on monday: {teams == np.sort(np.unique(teams_on_monday)).tolist()}"
    )

    return teams == np.sort(np.unique(teams_on_monday)).tolist()


if __name__ == "__main__":
    num_teams = 6
    weeks_between = 1
    profits = [
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
    ]

    t = 2 / 3
    s = 2

    sol = [
        [[{5, 1}], [{3, 6}], [{2, 4}], [{np.nan, np.nan}], [{np.nan, np.nan}]],
        [[{6, 2}], [{1, 4}], [{3, 5}], [{np.nan, np.nan}], [{np.nan, np.nan}]],
        [[{5, 4}], [{1, 6}], [{np.nan, np.nan}], [{5, 2}], [{np.nan, np.nan}]],
        [[{6, 4}], [{1, 3}], [{np.nan, np.nan}], [{5, 2}], [{np.nan, np.nan}]],
        [[{1, 2}], [{3, 4}], [{5, 6}], [{np.nan, np.nan}], [{np.nan, np.nan}]],
        [[{6, 3}], [{4, 1}], [{np.nan, np.nan}], [{2, 5}], [{np.nan, np.nan}]],
        [[{3, 1}], [{4, 2}], [{np.nan, np.nan}], [{6, 5}], [{np.nan, np.nan}]],
        [[{1, 5}], [{np.nan, np.nan}], [{np.nan, np.nan}], [{4, 3}], [{2, 6}]],
        [[{4, 5}], [{2, 3}], [{np.nan, np.nan}], [{6, 1}], [{np.nan, np.nan}]],
        [[{4, 6}], [{5, 3}], [{2, 1}], [{np.nan, np.nan}], [{np.nan, np.nan}]],
    ]

    validate(
        sol=sol,
        num_teams=num_teams,
        weeks_between=weeks_between,
        profits=profits,
        t=t,
        s=s,
    )
