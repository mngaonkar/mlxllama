import mlx.core as mx
import mlx.nn as nn
from models.args import ModelArgs 

class BaseModel(nn.Module):
    """Base model class for LLM."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

    def __call__(self, 
                 input: mx.array,
                 cache = None):
        """Forward pass."""
        return NotImplementedError("Model must implement __call__")

    def loss(self, 
            input: mx.array,
            target: mx.array,
            loss_mask: mx.array):
        """Compute loss."""
        logits = self.__call__(input)
        logits = logits.asdtype(mx.float32)

        losses = mx.losses.cross_entropy(logits, target) * loss_mask
        num_tokens = loss_mask.sum()
        value = losses.sum() / num_tokens

        return value, None, num_tokens
        
    
    @staticmethod    
    def create_additive_causal_mask(N: int, offset: int = 0):
        rinds = mx.arange(offset + N)
        linds = mx.arange(offset, offset + N) if offset else rinds
        mask = linds[:, None] < rinds[None]
        
        return mask * -1e9