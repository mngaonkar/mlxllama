import logging
import dataclasses

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class ModelArgs():
    """Model arguments."""
    model_type: str
    model_name: str
    model_path: str
    dim: int
    n_layers: int
    head_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float = None
    hidden_dim: int = None
    vocab_size: int = -1
    rope_theta: float = 10000.0
    rope_traditional: bool = True
    partial_rotary_factor: float = None
    hidden_act: str = None
    max_position_embeddings: int = 0
    original_max_position_embeddings: int = 0
    tie_word_embeddings: bool = False
    bos_token_id: int = None
    eos_token_id: int = None
    pad_token_id: int = None
    quantization: dict = None

    def __repr__(self) -> str:
        return str(dataclasses.asdict(self))

    @staticmethod
    def load_config(config):
        ArgClass = None

        logger.info(f"Loading model type: {config['model_type']}")
        if 'model_type' in config:
            if config['model_type'] in ['llama', 'mistral', 'gemma']:
                ArgClass = LlamaArgs

        if ArgClass is None:
            logger.warning(f"Model type {config['model_type']} not recognized. Using default LlamaArgs.")
            ArgClass = LlamaArgs

        config = {k:v for k,v in config.items() if k in ArgClass.__dataclass_fields__}

        return ArgClass(**config)

@dataclasses.dataclass
class LlamaArgs(ModelArgs):
    rope_scaling: dict = None
