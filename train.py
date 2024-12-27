from __future__ import absolute_import, division, print_function

from src.trainer import Trainer
from arg_parser import MonodepthOptions
from src.logger import WandBWriter

options = MonodepthOptions()
config = options.parse()
writer = WandBWriter()

if __name__ == "__main__":
    trainer = Trainer(writer, config)
    trainer.train()
