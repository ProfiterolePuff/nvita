import datetime
import json
import logging
import os
import random
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)

def set_seed(random_state=None):
    """Reset RNG seed."""
    if random_state is None:
        random_state = random.randint(1, 999999)
    random_state = int(random_state)
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True
    logger.info(f'Set random state to: {random_state}')
    return random_state

def time2str(time_elapsed, formatstr='%Hh%Mm%Ss'):
    """Format millisecond to string."""
    return time.strftime(formatstr, time.gmtime(time_elapsed))


def to_json(data_dict, path):
    """Save dictionary as JSON."""
    def converter(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.__str__()

    with open(path, 'w') as file:
        logger.info(f'Save to: {path}')
        json.dump(data_dict, file, default=converter)


def open_json(path):
    """Read JSON file."""
    try:
        with open(path, 'r') as file:
            data_json = json.load(file)
            return data_json
    except:
        logger.error(f'Cannot open {path}')


def create_dir(path):
    """Create directory if the input path is not found."""
    if not os.path.exists(path):
        logger.info(f'Creating directory: {path}')
        os.makedirs(path)
        
