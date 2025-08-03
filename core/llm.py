import logging
from models.args import ModelArgs
import models
import time
import mlx.core as mx
from core.tokenizer import Tokenizer
from models.llama import KVCache
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten
import models.llama
import numpy as np
from mlx_lm.sample_utils import apply_top_p

# from mlx_lm import sample_utils

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def top_k_sampling(logits, top_k: int):
    """Apply top-k sampling to logits."""
    # Get the top k logit values and indices
    top_logits, top_indices = mx.topk(logits, top_k, axis=-1)
    
    # Create a mask for values not in top-k
    mask = mx.full_like(logits, float('-inf'))
    mask = mask.at[..., top_indices].set(top_logits)
    
    return mask

def top_p_sampling(logits, top_p: float):
    """Apply nucleus (top-p) sampling to logits."""
    # Sort logits in descending order
    sorted_logits = mx.sort(logits, axis=-1)[:, ::-1]
    sorted_indices = mx.argsort(logits, axis=-1)[:, ::-1]
    
    # Convert to probabilities
    probs = mx.softmax(sorted_logits, axis=-1)
    
    # Calculate cumulative probabilities
    cumulative_probs = mx.cumsum(probs, axis=-1)
    
    # Find the cutoff point
    cutoff = cumulative_probs <= top_p
    
    # Keep at least one token
    cutoff = cutoff.at[:, 0].set(True)
    
    # Create mask for selected tokens
    mask = mx.full_like(logits, float('-inf'))
    selected_logits = mx.where(cutoff, sorted_logits, float('-inf'))
    
    # Scatter back to original positions
    for i in range(logits.shape[0]):
        mask = mask.at[i, sorted_indices[i]].set(selected_logits[i])
    
    return mask

class LLM():
    model_mapping = {
        "llama": models.llama.Model,
    }

    def __init__(self, 
                 tokenizer: Tokenizer,
                 args: ModelArgs) -> None:
        self.args = args
        self.model = self._get_model_class(args.model_type)(args)
        # logger.info(f"Model initialized: {self.model}")
        self.model.tokenizer = tokenizer
        self.tokenizer = tokenizer

    def _get_model_class(self, name: str):
        """Get model class."""
        if name in self.model_mapping:
            return self.model_mapping[name]
        else:
            raise ValueError(f"Model {name} not found.")
        
    def quantize(self,
                 group_size: int = 32,
                 bits: int = 4):
        """Quantize model."""
        assert self.model is not None, "Model is not initialized, checked before quantizing"
        nn.quantize(self.model, group_size=group_size, bits=bits)
        assert self.model is not None, "Model is not initialized, checked after quantizing"
        

    def verify_weights(self, 
                       weights: dict):
        """Verify weights."""
        if self.model is None:
            raise ValueError("Model is not initialized")
        model_params = tree_flatten(self.model.parameters())
        result = True

        # logger.info(f"GGUF weight keys: {weights.keys()}")
        for name, weight in model_params:
            if name not in weights:
                result = False
                logger.warning(f"MLX weight key {name} not found in GGUF weights.")
            elif weight.shape != weights[name].shape:
                result = False
                logger.warning(f"Shape mismatch for {name}: {weight.shape} != {weights[name].shape}")
        
        model_keys = {name for name, _ in model_params}
        for name in weights:
            if name not in model_keys:
                result = False
                logger.warning(f"GGUF weight key {name} not found in MLX model.")
        
        return result
    
    def get_size(self):
        """Get size."""
        if self.model is None:
            raise ValueError("Model is not initialized")
        pp_flat = tree_flatten(self.model.parameters()) # each element in the list is [name, tensor]
        params = sum([p[1].size for p in pp_flat])

        return params
    
    def update_weights(self,
                       weights: dict):
        """Update weights."""
        if self.model is None:
            raise ValueError("Model is not initialized")
        weights = tree_unflatten(list(weights.items()))
        self.model.update(weights)
        mx.eval(self.model.parameters())

    def completion(self, *args, **kwargs):
        """Completion."""
        response = ""

        for text, metadata in generate(self.model, self.tokenizer, *args, **kwargs):
            response += text

        return response, metadata    

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
    encoded = tokenizer.encode(prompt)
    logger.info(f"encoded: {encoded}")
    # if not encoded:
    #     raise ValueError("Failed to encode prompt or prompt is empty")
    inputs = mx.array(encoded)
    logger.info(f"input encoded length: {len(inputs)}")
    logger.info(f"input encoded: {inputs}")

    stop_tokens = tokenizer.special_ids
    if extra_stop_tokens:
        stop_tokens.extend(extra_stop_tokens)
    tokens, text = [], ""

    def sample(logits):
        """Sample."""
        if temperature == 0.0:
            y = mx.argmax(logits, axis=-1)
        else:
            logits = logits * (1.0 / temperature)

            if logit_filter is not None:
                logits = logit_filter(logits)
            
            if len(tokens) > 0 and repetition_penalty != 1.0:
                for token in tokens[-repetition_window:]:
                    logits[0, token] = logits[0, token] / repetition_penalty
            
            if top_k > 0:
                logits = mx.topk(logits, top_k)

            # TODO: fix top_p sampling
            # if top_p > 0.0:
            #     logits = mx.topk(logits, top_p)
            logits = apply_top_p(logits, top_p=top_p)

            y = mx.random.categorical(logits)

        p = 0.0
        if logprobes:
            p = nn.log_softmax(logits, axis=-1)[0,y].item()
        
        return y, p

    def generate_step(model: nn.Module, 
                      inputs: mx.array):
        logits = None

        if prompt_cache is not None:
            logits, cache = prompt_cache.get(inputs)
        
        if logits is None:
            logger.debug("Cache miss.")
            cache = KVCache(model.args.head_dim, model.args.n_kv_heads)
            logits = model(inputs[None], cache)
            # logger.debug(f"logits shape: {logits.shape}")
            # logger.debug(f"logits: {logits}")

            logits = logits[:, -1, :]

            if prompt_cache is not None:
                prompt_cache.put(inputs, logits, cache)
        y, p = sample(logits)
        yield y, p

        while True:
            logits = model(y[None], cache)
            # logger.debug(f"logits shape: {logits.shape}")
            # logger.debug(f"logits: {logits}")
            logits = logits[:, -1, :]

            y, p = sample(logits)
            yield y, p
    
    for (token, p), i in zip(generate_step(model, inputs), range(max_tokens)):
        # logger.debug(f"Token: {token.item()}, p: {p}, i: {i}")

        if i == 0:
            mx.eval(token)
        
        if token.item() in stop_tokens:
            break
        
        tokens.append(token.item())
        # logger.debug(f"Token appended: {token.item()}")
        if (len(tokens) % flush) == 0:
            mx.eval(tokens)
            text_offset = len(text)
            text = tokenizer.decode(tokens)

            yield text[text_offset:], None


    mx.async_eval(tokens)
    text_offset = len(text)
    text = tokenizer.decode(tokens)

    yield text[text_offset:], None

