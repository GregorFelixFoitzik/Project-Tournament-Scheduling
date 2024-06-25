import os
import yaml

with open(file="configs/run_config.yaml", mode="r") as file:
    run_config = yaml.safe_load(stream=file)

metaheuristics_to_use = run_config["metaheuristics"]

for metaheuristic_name in metaheuristics_to_use:
    print(f"Metaheuristic: {metaheuristic_name}")
    file_names = os.listdir(path="data")
    for file_name in file_names:
        path_to_file = f"data/{file_name}"
        print(f"\tFile-name: {path_to_file}")

        os.system(command=f"python main.py {path_to_file} {metaheuristic_name}")
