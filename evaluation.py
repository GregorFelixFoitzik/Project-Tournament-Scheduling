print(metaheuristics_to_use, run_config)

def get_yaml_config(metaheuristic_name: str) -> dict[str, Union[int, float]]:
    """Load the yaml-config file for a given metaheuristic.

    Args:
        metaheuristic_name (str): Name of the metaheuristic.

    Returns:
        dict[str, Union[int, float]]: Dictionary containing the configuration.
    """
    with open(file=f"configs/{metaheuristic_name}.yaml", mode="r") as file:
        config = yaml.safe_load(stream=file)

    return config["parameters"]


parameters = get_yaml_config(metaheuristic_name='tabu_search_lns_config_16')

print(parameters)