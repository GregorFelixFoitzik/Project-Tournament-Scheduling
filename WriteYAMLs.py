import yaml

# for i, value in enumerate(max_size_values):
#     # Create the data structure
#     data = {
#         'parameters': {
#             'max_size_tabu_list': value
#         }
#     }
#     # File path for each YAML file
#     file_path = f'tabu_search_{i+1}.yaml'
    
#     # Writing the data to a YAML file
#     with open(file_path, 'w') as file:
#         yaml.dump(data, file, default_flow_style=False)


def create_configs(metaheuristic: str, parameters: dict):
    for ii, config in enumerate(parameters):
        print(config)

parameter = {
    'parameter': {
        'max_size_values' : [vv for vv in range(5, 51, 5)]
    }
}

print(
    create_configs('tabu_search', parameters=parameter)
)