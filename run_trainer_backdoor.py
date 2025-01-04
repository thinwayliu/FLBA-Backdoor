# -*- coding: utf-8 -*-
import sys

sys.dont_write_bytecode = True

from core.config import Config
from core import Poison_Trainer

if __name__ == "__main__":
    config = Config("./config/baseline.yaml").get_config_dict()
    trainer = Poison_Trainer(config)
    trainer.train_loop()
