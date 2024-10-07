import logging
from models.args import ModelArgs
import models
import time
import mlx.core as mx
from core.tokenizer import Tokenizer
from mlx_lm.models.base import KVCache
import mlx.nn as nn
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import models.llama
import numpy as np

logger = logging.getLogger(__name__)

class LLM():
    model_mapping = {
        "llama": models.llama.Model,
    }

    def __init__(self, 
                 tokenizer: Tokenizer,
                 args: ModelArgs) -> None:
        self.args = args
        self.model = self._get_model_class(args.model_type)(args)
        self.tokenizer = tokenizer

    def _get_model_class(self, name: str):
        """Get model class."""
        if name in self.model_mapping:
            return self.model_mapping[name]
        else:
            raise ValueError(f"Model {name} not found.")
        
    def quantize(self,
                 group_size: int = 32,
                 bits: int = 4,
                 weights:dict = None):
        """Quantize model."""
        self.model = nn.quantize(self.model, group_size=group_size, bits=bits, weights=weights)

    def verify_weights(self, 
                       weights: dict):
        """Verify weights."""
        model_params = tree_flatten(self.model.parameters())
        result = True

        for name, weight in model_params:
            if name not in weights:
                result = False
                logger.warning(f"Weight {name} not found in weights.")
            elif weight.shape != weights[name].shape:
                result = False
                logger.warning(f"Shape mismatch for {name}: {weight.shape} != {weights[name].shape}")
        
        model_keys = {name for name, _ in model_params}
        for name in weights:
            if name not in model_keys:
                result = False
                logger.warning(f"Weight {name} not found in model.")
        
        return result
    
    def get_size(self):
        """Get size."""
        pp_flat = tree_flatten(self.model.parameters()) # each element in the list is [name, tensor]
        params = sum([p[1].size for p in pp_flat])

        return params
    
    def update_weights(self,
                       weights: dict,
                       mapping: dict = None):
        """Update weights."""
        weights = tree_unflatten(list(weights.items()))
        self.model.update(weights)
        mx.eval(self.model.parameters())

def generate(model,
            tokenizer: Tokenizer,
            prompt: str,
            max_tokens: int = 2048,
            temperature: float = 0.0,
            top_k: int = 0,
            top_p: float = 0.9,
            repetition_penalty: float = 1.0,
            repetition_window: int = 25,
            logprobes: bool = False,
            token_ids: bool = False,
            flush: int = 5,
            extra_stop_tokens: list = None,
            prompt_cache = None,
            logit_filter = None,
            ):
    """Generate."""
    start = time.perf_counter()
    logger.info("Generating text...")
    inputs = mx.array(tokenizer.encode(prompt))

    stop_tokens = tokenizer.special_ids
    tokens, text = [], ""

    def sample(logits):
        if temperature == 0.0:
            y = mx.argmax(logits, axis=-1)
        else:
            logits = logits * (1.0 / temperature)

            if logit_filter is not None:
                logits = logit_filter(logits)
            
            if len(tokens) > 0 and repetition_penalty is not None:
                logits = logits * repetition_penalty
            
            if top_k > 0:
                logits = mx.top_k(logits, k=top_k)

            if top_p > 0.0:
                logits = mx.top_p(logits, p=top_p)

            y = mx.random.categorical(logits, 1)

        p = 0.0
        if logprobes:
            p = nn.log_softmax(logits, axis=-1)[0,y].item()
        
        return y, p

    def generate_step(model, inputs):
        logits = None

        if prompt_cache is not None:
            logits, cache = prompt_cache.get(inputs)
        
        if logits is None:
            cache = KVCache.for_model(model)
            logits = model(inputs[None], cache)
            logits = logits[:, -1, :]

            if prompt_cache is not None:
                prompt_cache.put(inputs, logits, cache)
        y, p = sample(logits)
        yield y, p

        while True:
            logits = model(y[None], cache)
            logits = logits[:, -1, :]

            y, p = sample(logits)
            yield y, p
    
    for (token, p), i in zip(generate_step(model, inputs), range(max_tokens)):
        if i == 0:
            mx.eval(token)
        
        if token.item() in stop_tokens:
            break
        
        tokens.append(token.item())
        if (len(tokens) % flush) == 0:
            mx.eval(tokens)
            text_offset = len(text)
            text = tokenizer.decode(tokens)

            yield text[text_offset:], None


    mx.async_eval(tokens)
    text_offset = len(text)
    text = tokenizer.decode(tokens)

    yield text[text_offset:], None





