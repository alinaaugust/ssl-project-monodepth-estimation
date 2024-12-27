from __future__ import absolute_import, division, print_function
import torch

from src.trainer import Trainer
from arg_parser import MonodepthOptions
from src.logger import WandBWriter
torch.backends.cudnn.benchmark = True

options = MonodepthOptions()
config = options.parse()
writer = WandBWriter(config)

if __name__ == "__main__":
    trainer = Trainer(writer, config)
    trainer.train()
