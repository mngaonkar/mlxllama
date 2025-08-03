import mlx.core as mx
from models.base import ModelArgs
import enum
from core.llm import LLM
import pathlib
import logging
from core.mapping import map_config, map_key
from core.tokenizer import GGUFTokenizer

logging.basicConfig(level=logging.INFO)
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

def load_model_file(model_path: str) -> LLM:
    """Load model from file."""
    model_path = pathlib.Path(model_path)

    if model_path.suffix == ".gguf":
        return load_gguf_file(str(model_path))
    else:
        raise ValueError(f"Model format {model_path.suffix} not supported.")
    
def load_gguf_file(model_path: pathlib.Path) -> LLM:
    logger.info(f"Loading GGUF model from {model_path}...")
    gguf_weights, metadata = mx.load(model_path, return_metadata=True)
    logger.info(f"metadata: {metadata.keys()}")

    config = map_config(metadata)
    config['model_path'] = str(model_path)
    logger.info(f"Config: {config}")
    model_args = ModelArgs.load_config(config)

    weights = {}
    mapping = {}

    # Map keys
    logger.info("Mapping keys...")
    for k, v in gguf_weights.items():
        mlx_key = map_key(k)
        mapping[mlx_key] = k    

        if mlx_key is None:
            logger.warning(f"Skipping key {k}")
        else:
            weights[mlx_key] = v

    # Print mapped keys
    for key, value in mapping.items():
        logger.info(f"Mapped keys: MLX key = {key} GGUF key = {value}")

    # Load quantization
    gguf_file_type = metadata.get("general.file_type", "unknown")
    logger.info(f"GGUF file type: {gguf_file_type}")

    quantization = None
    if gguf_file_type == 0 or gguf_file_type == 1:
        logger.info("No quantization found.")
    elif gguf_file_type == 2 or gguf_file_type == 3:
        logger.info("Quantization found.")
        quantization = {"group_size": 32, "bits": 4}
    elif gguf_file_type == 7:
        logger.info("Quantization found.")
        quantization = {"group_size": 32, "bits": 8}
    else:
        logger.warning(f"Unknown file type {gguf_file_type}")

    model_args.quantization = quantization

    tokenizer = GGUFTokenizer(metadata)
    logger.info(f"Tokenizer loaded from {model_path}")

    model = LLM(tokenizer, model_args)
    logger.info(f"Model loaded with {model_args}")

    if quantization is not None:
        logger.info(f'Quantizing model group_size = {quantization["group_size"]} bits = {quantization["bits"]}')
        model.quantize(group_size=quantization["group_size"], bits=quantization["bits"])

    model.verify_weights(weights)
    model.update_weights(weights)

    total_params = model.get_size()
    logger.info(f"Model loaded with {total_params/10**9:.2f}B parameters.")

    return model
