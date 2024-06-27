# Script to execute the algorithms via the console
import os
import yaml

from tqdm import tqdm

with open(file="configs//templates/run_config.yaml", mode="r") as file:
    run_config = yaml.safe_load(stream=file)

metaheuristics_to_use = run_config["metaheuristics"]
instances = ['data/'+f for f in os.listdir('data/') if f.endswith('.in')]
config_files = [f.split('.')[0] for f in os.listdir('configs/') if f.endswith('.yaml')]
timeout = 30


for mh in tqdm(metaheuristics_to_use[:], desc=f"Overall"):
    meta_config_files = [f for f in config_files if mh == f.split('---')[0]]
    
    for instance in tqdm(instances[:], desc=f"MH {mh}"):
         
        for config in tqdm(meta_config_files[:], desc="Config"):
            # print(mh,"\t",instance,"\t",config)
            os.system(command=f"python run_single_config.py {instance} {mh} {config} {timeout}")
