
import sys
sys.path.append('.')

from utils import load_config


def test_load_config():
    config_dct = load_config("tests/test.ini")
    assert isinstance(config_dct['test']['INT'], int)
    assert isinstance(config_dct['test']['FLOAT'], float)
    assert isinstance(config_dct['test']['LIST'], list)
    assert isinstance(config_dct['test']['STR'], str)

    assert isinstance(config_dct['test2']['INT_FROM_A'], int)
    assert config_dct['test2']['NONE'] is None
