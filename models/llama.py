from typing import Optional
import mlx.core as mx
import mlx.nn as nn
from models.args import ModelArgs
from models.activation import init_activation
from models.rope import init_rope
from models.base import BaseModel
from sentencepiece import SentencePieceProcessor
import logging

logger = logging.getLogger(__name__)

class KVCache:
    def __init__(self, head_dim: int, n_kv_heads: int):
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.keys = None
        self.values = None
        self.offset = 0
        
    def update_and_fetch(self, keys, values):
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=-2)
            self.values = mx.concatenate([self.values, values], axis=-2)
        
        self.offset += keys.shape[-2]
        return self.keys, self.values

class Model(BaseModel):
    """Model class for LLM."""
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.args = args
        self.n_layers = args.n_layers
        self.vocab_size = args.vocab_size
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = [TransformerBlock(args=args) for _ in range(args.n_layers)]
        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.model_path = args.model_path
        self.tokenizer = SentencePieceProcessor(model_file=str(self.model_path / "tokenizer.model"))

        if args.tie_word_embeddings:
            self.output = None
        else:
            self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def __call__(self, 
                 input: mx.array,
                 cache = None):
        """Forward pass."""
        h = self.tok_embeddings(input)
        mask = None

        if h.shape[1] > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(h.shape[1]).astype(h.dtype)

        if cache is None:
            cache = [None] * self.n_layers

        logger.info(f"Cache: {cache}")
        for i, layer in enumerate(self.layers):
            h = layer.forward(h, mask, cache[i])
        
        if self.output is None:
            return self.tok_embeddings.as_linear(self.norm(h))
        else:
            return self.output(self.norm(h))



class Attention(nn.Module):
    """Attention layer."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.scale = self.args.head_dim ** -0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

        self.rope = init_rope(args)

    def __call__(self, 
                 input: mx.array,
                 mask: Optional[mx.array] = None,
                 cache: Optional[KVCache] = None) -> mx.array:
        """Forward pass."""
        mx.eval(input)
        logger.info(f"input shape: {input.shape}")
        B, L, D = input.shape

        queries, keys, values = self.wq(input), self.wk(input), self.wv(input)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            if cache.offset > 0 and L > 1:
                mask = BaseModel.create_additive_causal_mask(L, offset=cache.offset)
                
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale, mask=mask)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.wo(output)

class FeedForward(nn.Module):
    """Feed-forward layer."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=False)

        self.act = init_activation(args)
        

    def __call__(self, 
                 input: mx.array) -> mx.array:
        """Forward pass."""
        return self.w2(self.act(self.w1(input)) * self.w3(input))

class TransformerBlock(nn.Module):
    """Transformer block."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.dim = args.dim

        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ff_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, 
                 input: mx.array,
                 mask: Optional[mx.array] = None,
                 cache: Optional[KVCache] = None):
        """Forward pass."""
        r = self.attention(self.attention_norm(input), mask, cache)
        h = input + r
        r = self.feed_forward(self.ff_norm(h))
        out = h + r

        return out