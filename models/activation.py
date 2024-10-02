import mlx.core as mx
import mlx.nn as nn
from models.args import ModelArgs
import logging
from functools import partial

logger = logging.getLogger(__name__)

@partial(mx.compile, shapless=True)
def relu2(x):
    return nn.relu(x).square()

def init_activation(args: ModelArgs):
    """Initialize activation function."""

    if args.hidden_act is None:
        return nn.silu
    
    if args.hidden_act == "relu":
        return nn.relu
    elif args.hidden_act == "gelu":
        return nn.gelu
    elif args.hidden_act == "tanh":
        return nn.tanh
    elif args.hidden_act == "sigmoid":
        return nn.sigmoid
    elif args.hidden_act == "silu":
        return nn.silu
    elif args.hidden_act == "relu2":
        return relu2
    else:
        return nn.silu
