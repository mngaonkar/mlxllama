import logging
import collections
import hashlib
import mlx.core as mx
from mlx_lm.models.base import KVCache
from typing import Optional

logger = logging.getLogger(__name__)

class PromptCache():
    """Prompt cache."""
    def __init__(self,
                 max_size: int = 10):
        self.logits = {}
        self.kv_cache = {}
        self.lru = collections.OrderedDict()
        self.max_size = max_size

    def _key(self, 
             inputs: mx.array):
        """Key."""
        return hashlib.sha256(inputs).hexdigest()

    def get(self, 
            inputs: mx.array):
        """Get."""
        if type(inputs) is mx.array:
            key = self._key(inputs)
        else:
            key = inputs

        if key in self.lru:
            self.lru.move_to_end(key)
            self.lru[key] += 1

            logits = self.logits[key]
            kv_cache = [c.copy() for c in self.kv_cache[key]]

            return logits, kv_cache

        return None, None

    def put(self, 
            inputs: mx.array, 
            logits: mx.array,
            kv_cache: Optional[KVCache] = None):
        """Set."""
        key = self._key(inputs)
        if key not in self.lru:
            self.logits[key] = logits
            self.kv_cache[key] = [c.copy() for c in kv_cache]
            self.lru[key] = 0
        
        if len(self.lru) > self.max_size:
            pop_key, _ = self.lru.popitem(last=False)
            self.logits.pop(pop_key)
            self.kv_cache.pop(pop_key)

    