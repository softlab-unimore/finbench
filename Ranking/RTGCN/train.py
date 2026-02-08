#!/usr/bin/env python
import argparse
import random
import sys

import numpy as np
import torch

# torchlight
from torchlight import import_class


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')
    parser.add_argument('--seed', type=int, default=42, help='random seed for initialization')
    parser.add_argument('--processor', default='recognition', help='the processor will be used')

    # region register processor yapf: disable
    processors = dict()
    processors['recognition'] = import_class('processor.recognition.REC_Processor')
    #endregion yapf: enable

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()
    set_seed(arg.seed)

    # start
    Processor = processors[arg.processor]
    p = Processor(sys.argv[4:])

    p.start()
