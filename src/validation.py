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
    feasible_check_matches = check_matches(sol=sol, num_teams=num_teams)
    feasible_each_team_monday = each_team_on_monday(sol=sol, num_teams=num_teams)

    return feasible_uniqueness and feasible_check_matches and feasible_each_team_monday


def uniqueness(sol: list[list[list[str]]], num_teams: int, weeks: int) -> bool:
    # Source: https://www.pythonpool.com/flatten-list-python/, accessed 29.05.2023
    matches = np.unique(ar=list(itertools.chain(*list(itertools.chain(*sol)))))
    print(
        f"Uniqueness ({np.shape(a=matches)[0] - 1} and {int((num_teams/2)*weeks)}):"
        f"{np.shape(a=matches)[0] - 1 == ((num_teams/2)*weeks)}"
    )

    return np.shape(a=matches)[0] == num_teams


def check_matches(sol: list[list[list[str]]], num_teams: int) -> bool:
    matches_required = np.array(
        object=list(
            set(
                [
                    f"{i}vs{j}" if i != j else "-"
                    for j in range(num_teams)
                    for i in range(num_teams)
                ]
            )
        )
    )
    matches_required = matches_required[matches_required != "-"]
    matches_required = np.sort(a=matches_required)

    # Source: https://www.pythonpool.com/flatten-list-python/, accessed 29.05.2023
    sol_flattend = np.array(object=list(itertools.chain(*list(itertools.chain(*sol)))))
    sol_flattend = sol[sol != "-"]
    sol_flattend = np.sort(a=matches_required)

    print(f"XvsY and YvsX: {np.unique(matches_required == sol_flattend)[0]}")

    return np.unique(matches_required == sol_flattend)[0]


def each_team_on_monday(sol: list[list[list[str]]], num_teams: int) -> bool:
    teams = [str(team) for team in range(1, num_teams + 1)]
    teams_on_monday = []
    for matches_monday in sol:
        for match in matches_monday[0]:
            if match != "-":
                teams_on_monday += match.split("vs")

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
        [["5vs1"], ["3vs6"], ["2vs4"], ["-"], ["-"]],
        [["6vs2"], ["1vs4"], ["3vs5"], ["-"], ["-"]],
        [["5vs4"], ["1vs6"], ["-"], ["3vs2"], ["-"]],
        [["6vs4"], ["1vs3"], ["-"], ["5vs2"], ["-"]],
        [["1vs2"], ["3vs4"], ["5vs6"], ["-"], ["-"]],
        [["6vs3"], ["4vs1"], ["-"], ["2vs5"], ["-"]],
        [["3vs1"], ["4vs2"], ["-"], ["6vs5"], ["-"]],
        [["1vs5"], ["-"], ["-"], ["4vs3"], ["2vs6"]],
        [["4vs5"], ["2vs3"], ["-"], ["6vs1"], ["-"]],
        [["4vs6"], ["5vs3"], ["2vs1"], ["-"], ["-"]],
    ]

    validate(
        sol=sol,
        num_teams=num_teams,
        weeks_between=weeks_between,
        profits=profits,
        t=t,
        s=s,
    )
