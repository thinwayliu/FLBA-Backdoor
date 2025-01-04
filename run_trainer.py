# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

from core.config import Config
from core import Trainer


if __name__ == "__main__":
    config = Config("./config/reproduce/Proto/ProtoNet-miniImageNet--ravi-resnet12-5-5-Table2.yaml").get_config_dict()
    trainer = Trainer(config)
    trainer.train_loop()
