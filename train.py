from __future__ import absolute_import, division, print_function

from src.trainer import Trainer
from arg_parser import MonodepthOptions

options = MonodepthOptions()
config = options.parse()


if __name__ == "__main__":
    trainer = Trainer(config)
    trainer.train()
