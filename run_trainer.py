# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

from core.config import Config
from core import Trainer


if __name__ == "__main__":
    config = Config("./config/reproduce/Baseline++/BaselinePlus-tiered_imagenet-resnet12-Table2.yaml").get_config_dict()
    trainer = Trainer(config)
    trainer.train_loop()
