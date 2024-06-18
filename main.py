# Standard library
import argparse
import numpy as np

# Project specific library
import yaml

from src.helper import read_in_file
from src.greedy_heuristic import GreedyHeuristic
from src.metaheuristics.metaheuristics_controller import main_metaheuristics_controller

# Copied from https://docs.python.org/3/library/argparse.html, accessed 28.05.2024
parser = argparse.ArgumentParser(description="Input for algorithms")
# ChatGPT fixed the issues that default values were not used. => python main.py works now
parser.add_argument(
    "-t", "--timeout", type=int, default=30, help="Timeout duration in seconds"
)
parser.add_argument(
    "-p",
    "--path_to_instance",
    type=str,
    default="data/example.in",
    help="Path to the input file",
)


def main():
    with open(file="configs/run_config.yaml", mode="r") as file:
        run_config = yaml.safe_load(stream=file)

    metaheuristics_to_use = run_config["metaheuristics"]

    num_teams = 6

    sol = np.array(
        [
            [
                [[5, 1], [np.nan, np.nan]],
                [[3, 6], [2, 4]],
                [[np.nan, np.nan], [np.nan, np.nan]],
            ],
            [
                [[6, 2], [np.nan, np.nan]],
                [[1, 4], [3, 5]],
                [[np.nan, np.nan], [np.nan, np.nan]],
            ],
            [
                [[5, 4], [np.nan, np.nan]],
                [[1, 6], [np.nan, np.nan]],
                [[3, 2], [np.nan, np.nan]],
            ],
            [
                [[6, 4], [np.nan, np.nan]],
                [[1, 3], [np.nan, np.nan]],
                [[5, 2], [np.nan, np.nan]],
            ],
            [
                [[1, 2], [np.nan, np.nan]],
                [[3, 4], [5, 6]],
                [[np.nan, np.nan], [np.nan, np.nan]],
            ],
            [
                [[6, 3], [np.nan, np.nan]],
                [[4, 1], [np.nan, np.nan]],
                [[2, 5], [np.nan, np.nan]],
            ],
            [
                [[3, 1], [np.nan, np.nan]],
                [[4, 2], [np.nan, np.nan]],
                [[6, 5], [np.nan, np.nan]],
            ],
            [
                [[1, 5], [np.nan, np.nan]],
                [[4, 3], [2, 6]],
                [[np.nan, np.nan], [np.nan, np.nan]],
            ],
            [
                [[4, 5], [np.nan, np.nan]],
                [[2, 3], [np.nan, np.nan]],
                [[6, 1], [np.nan, np.nan]],
            ],
            [
                [[4, 6], [np.nan, np.nan]],
                [[5, 3], [2, 1]],
                [[np.nan, np.nan], [np.nan, np.nan]],
            ],
        ]
    )
    # sol = np.array(
    #     [
    #         [
    #             [[5, 1], [np.nan, np.nan]],
    #             [[3, 2], [np.nan, np.nan]],
    #             [[6, 4], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[6, 2], [np.nan, np.nan]],
    #             [[1, 4], [np.nan, np.nan]],
    #             [[3, 5], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[1, 6], [np.nan, np.nan]],
    #             [[2, 4], [np.nan, np.nan]],
    #             [[5, 3], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[1, 3], [np.nan, np.nan]],
    #             [[5, 2], [np.nan, np.nan]],
    #             [[4, 6], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[1, 2], [np.nan, np.nan]],
    #             [[3, 4], [np.nan, np.nan]],
    #             [[5, 6], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[4, 1], [np.nan, np.nan]],
    #             [[2, 5], [np.nan, np.nan]],
    #             [[6, 3], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[1, 5], [np.nan, np.nan]],
    #             [[4, 2], [np.nan, np.nan]],
    #             [[3, 6], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[3, 1], [np.nan, np.nan]],
    #             [[2, 6], [np.nan, np.nan]],
    #             [[4, 5], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[6, 1], [np.nan, np.nan]],
    #             [[2, 3], [np.nan, np.nan]],
    #             [[5, 4], [np.nan, np.nan]],
    #         ],
    #         [
    #             [[2, 1], [np.nan, np.nan]],
    #             [[4, 3], [np.nan, np.nan]],
    #             [[6, 5], [np.nan, np.nan]],
    #         ],
    #     ]
    # )

    algo_config = {
        "n": num_teams,
        "t": 2 / 3,
        "s": 2,
        "r": 3,
        "p": np.array(
            [
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
        ),
    }
    main_metaheuristics_controller(
        start_sol=sol,
        metaheuristics_to_use=metaheuristics_to_use,
        algo_config=algo_config,
        time_out=4,
    )


if __name__ == "__main__":
    main()
    exit(-1)
    # Extract command line arguments
    args = parser.parse_args()
    path_to_file = args.path_to_instance
    timeout = args.timeout

    algo_config = read_in_file(path_to_file=path_to_file)
    # print(
    #     algo_config['p'].reshape(3,6,6)
    # )
    agent = GreedyHeuristic(algo_config)
    agent.execute_cmd()
