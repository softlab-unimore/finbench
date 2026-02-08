
import math
from dataclasses import dataclass
from typing import Union, List, Dict, Any

@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    seq_in: int
    seq_out: int
    d_state: int =128
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 3
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)

        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

