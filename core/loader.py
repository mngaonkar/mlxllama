import mlx.core as mx
from models.base import ModelArgs
import enum
from core.llm import LLM
import pathlib
import logging

logger = logging.getLogger(__name__)

class ModelFormat(enum.Enum):
    UNKNOWN = 0
    MLX = 1
    GGUF = 2
    HUGGINGFACE = 3

    @staticmethod
    def guess_from_weights(weights: dict):
        for k in weights:
            if k.startswith("layers."):
                return ModelFormat.MLX
            elif k.startswith("blk."):
                return ModelFormat.GGUF
            elif k.startswith("transformer.") or k.startswith("model.layers."):
                return ModelFormat.HUGGINGFACE
        
        return ModelFormat.UNKNOWN
    
def load(model_path: str) -> LLM:
    """Load model."""
    model_path = pathlib.Path(model_path)

    if model_path.is_dir():
        return load_model_dir(model_path)
    elif model_path.is_file():
        return load_model_file(model_path)
    else:
        raise ValueError(f"Model path {model_path} not found.")


def load_model_dir(model_path: pathlib.Path) -> LLM:
    """Load model from directory."""
    pass

def load_model_file(model_path: pathlib.Path) -> LLM:
    """Load model from file."""
    model_path = pathlib.Path(model_path)

    if model_path.suffix == "gguf":
        return load_gguf_file(model_path)
    else:
        raise ValueError(f"Model format {model_path.suffix} not supported.")
    
def load_gguf_file(model_path: pathlib.Path) -> LLM:
    logger.info(f"Loading GGUF model from {model_path}...")
    gguf_weigthts, metadata = mx.load(model_path, return_metadata=True)

    weights = {}
    mapping = {}

    for k, v in gguf_weigthts.items():
        mlx_key = map_key(k)
