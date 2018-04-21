import json
import os 
from bunch import Bunch
import time


def get_config_from_json(json_file):

    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)


    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    config.tensorboard_log_dir = os.path.join("experiments", time.strftime("%Y-%m-%d/", time.localtime()), config.exp_name, "logs/")
    config.checkpoint_dir = os.path.join("experiments", time.strftime("%Y-%m-%d/", time.localtime()), config.exp_name, "checkpoints/")
    return config
