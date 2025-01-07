# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

import os

from core.config import Config
from core import Poison_Test

PATH = "./results/BaselinePlus-miniImageNet--ravi-resnet12-5-5-Dec-28-2022-13-18-47"
VAR_DICT = {
    "test_epoch": 5,
    "device_ids": "0,1",
    "n_gpu":2 ,
    "test_episode": 600,
    "episode_size": 1,
    "test_way": 5,
    "test_shot": 1,
    # 'shot_num': 3,
}

if __name__ == "__main__":
    config = Config(os.path.join(PATH, "config.yaml"), VAR_DICT).get_config_dict()
    test = Poison_Test(config, PATH)
    test.test_loop()
