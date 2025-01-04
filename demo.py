# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

import os

from core.config import Config
from core import Test

PATH = "./results/ProtoNet-miniImageNet--ravi-resnet12-5-5-Jul-22-2022-10-41-22"
VAR_DICT = {
    "test_epoch": 10,
    "device_ids": "0",
    "n_gpu": 1,
    "test_episode": 600,
    "episode_size": 1,
    "test_way": 5,
}

if __name__ == "__main__":
    config = Config(os.path.join(PATH, "config.yaml"), VAR_DICT).get_config_dict()
    test = Test(config, PATH)
    test.uap()
