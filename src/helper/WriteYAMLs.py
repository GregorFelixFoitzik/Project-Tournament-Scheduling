import yaml
from itertools import product

# Definiere die Parameterbereiche für jeden Algorithmus
param_grid = {
    # 'tabu_search': {
    #     'max_size_tabu_list': [ii for ii in range(1, 11, 2)],
    #     'neighborhood': ["random_swap_within_week", "select_worst_n_weeks"],
    # },
    # 'tabu_search_lns': {
    #     'max_size_tabu_list': [ii for ii in range(1, 11, 2)],
    # },
    'simulated_annealing': {
        'alpha': [0.85, 0.95],
        'temperature': [10000, 20000],
        'epsilon': [0.001, 0.0001],
        'neighborhood': ["random_swap_within_week", "select_worst_n_weeks"]
    },
    # 'lns_ts_simulated_annealing': {
    #     'alpha': [0.85, 0.95],
    #     'temperature': [10000, 20000],
    #     'epsilon': [0.001, 0.0001],
    #     'max_size_tabu_list': [ii for ii in range(1, 11, 2)],
    # },
    # 'large_neighborhood_search_simulated_annealing': {
    #     'alpha': [0.85, 0.95],
    #     'temperature': [10000, 20000],
    #     'epsilon': [0.001, 0.0001],
    # },
    # "reactive_tabu_search": {
    #     "rep": [3, 6],
    #     "chaos": [3, 6],
    #     "increase": [1.1, 1.5],
    #     "decrease": [0.8, 0.6],
    #     'neighborhood': ["select_worst_n_weeks", 'random_swap_within_week'],
    #     "cycle_max": [5, 10],
    # }
}

# Hilfsfunktion, um alle Kombinationen von Parametern zu erzeugen
def create_configs(param_dict):
    keys = param_dict.keys()
    values = param_dict.values()
    for combination in product(*values):
        yield dict(zip(keys, combination))

# Erstelle YAML-Dateien für jede Kombination von Parametern
for algorithm, params in param_grid.items():
    for i, config in enumerate(create_configs(params), start=1):
        data = {
            'parameters': config
        }
        # Datei-Name für jede YAML-Datei
        file_path = f'configs/{algorithm}---config_{i}.yaml'
        
        # Schreibe die Daten in eine YAML-Datei
        with open(file_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)

print("YAML files have been created successfully.")
