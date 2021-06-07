
from typing import Dict, Any
from configparser import ConfigParser, ExtendedInterpolation

from numpy import random as np_random
import torch
import random


def set_seed(seed):
    random.seed(seed)
    np_random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


def to_cuda(data):
    if isinstance(data, tuple):
        return [d.cuda() for d in data]
    elif isinstance(data, torch.Tensor):
        return data.cuda()
    raise RuntimeError


def load_config(config_file: str) -> dict:
    '''
    config example:
        # This is a comment
        [section]  # section
        a = 5  # int
        b = 3.1415  # float
        s = 'abc'  # str
        lst = [3, 4, 5]  # list
    
    it will ouput:
        (dict) {'section': {'a': 5, 'b':3.1415, 's': 'abc', 'lst':[3, 4, 5]}
    '''
    config = ConfigParser(interpolation=ExtendedInterpolation())
    # fix the problem of automatic lowercase
    config.optionxform = lambda option: option  # type: ignore
    config.read(config_file)

    config_dct: Dict[str, Dict] = dict()
    for section in config.sections():
        tmp_dct: Dict[str, Any] = dict()

        for key, value in config.items(section):
            if value == '':  # allow no value
                tmp_dct[key] = None
                continue
            try:
                tmp_dct[key] = eval(value)  # It may be unsafe
            except NameError:
                print("Note the configuration file format!")

        config_dct[section] = tmp_dct
    
    return config_dct
