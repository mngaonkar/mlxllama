import mlx.core as mx
import re

def map_key(key: str) -> str:
    if key.startswith("layers."):
        return key
    elif key.startswith("output."):
        return key
    elif key in ["tok_embeddings.weight", "norm.weight", "rope.freqs"]:
        return key
    elif key.startswith("model.embed_tokens."):
        return re.sub(r"^model\.embed_tokens\.", "tok_embeddings.", key)
    elif key.startswith("model.norm."):
        return re.sub(r"^model\.norm\.", "norm.", key)
    elif key.startswith("model.final_layernorm."):
        return re.sub(r"^model\.final_layernorm\.", "norm.", key)
    elif key.startswith("lm_head."):
        return re.sub(r"^lm_head\.", "output.", key)
    elif key.startswith("model.layers."):
        layer = key.split(".")[2]

        key = re.sub(r"^model\.layers\.", "layers.", key)
        key = re.sub(r"\.input_layernorm\.", ".attention_norm.", key)
        key = re.sub(r"\.post_attention_layer_norm\.", ".ffn_norm.", key)    
        key = re.sub(r"\.self_attn\.(k|v|q|o)_proj\.", r".attention.w\1.", key)
        key = re.sub(r"\.self_attn\.(k|q)_norm\.", r".attention.\1_norm.", key)
        key = re.sub(r"\.mlp\.gate_proj\.", ".feed_forward.w1.", key)
        key = re.sub(r"\.mlp\.down_proj\.", ".feed_forward.w2.", key)
        key = re.sub(r"\.mlp\.up_proj\.", ".feed_forward.w3.", key)

        return key
    elif key.startswith("output_norm."):
        return re.sub(r"^output_norm\.", "norm.", key)
    elif key.startswith("token_embed."):
        return re.sub(r"^token_embed\.", "tok_embeddings.", key)
    elif key.startswith("blk."):
        layer = key.split(".")[1]
        key = re.sub(r"^blk\.", "layers.", key)

        if key.endswith(".attn_norm.weight"):
            return f"layers.{layer}.attention_norm.weight"
        elif key.endswith(".ffn_norm.weight"):
            return f"layers.{layer}.ffn_norm.weight"
        
        key = re.sub(r"\.attn\.(k|v|q)\.", r".attention.w\1.", key)
        key = re.sub(r"\.attn_output\.", r".attention.wo.", key)

        return key

    return None

def map_keys(keys: list) -> list:
    return [map_key(k) for k in keys]

def map_config(config):
    result = {}

    mlx_keys = [
        "model_type",
        "dim",
        "n_layers",
        "n_heads",
        "head_dim",
        "hidden_dim",
        "n_kv_heads",
        "norm_eps",
        "vocab_size",
        "max_position_embeddings",
        "original_max_position_embeddings",
        "rope_theta",
        "rope_scaling",
        "partial_rotary_factor",
        "hidden_act",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "quantization",
        "moe",
        "tie_word_embeddings",
        "clip_qkv",
        "use_qk_norm",
        "logit_scale",
    ]

    for key in mlx_keys:
        if key in config:
            result[key] = config[key]
        
        key_map = {
        "hidden_size": "dim",
        "num_hidden_layers": "n_layers",
        "num_attention_heads": "n_heads",
        "intermediate_size": "hidden_dim",
        "num_key_value_heads": "n_kv_heads",
        "n_heads": "n_kv_heads",
        "rms_norm_eps": "norm_eps",
        "layer_norm_eps": "norm_eps",
        "norm_epsilon": "norm_eps",
        "layer_norm_bias": "norm_bias",
        "hidden_activation": "hidden_act",
        # GGUF metadata: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
        "llama.embedding_length": "dim",
        "llama.block_count": "n_layers",
        "llama.attention.head_count": "n_heads",
        "llama.feed_forward_length": "hidden_dim",
        "llama.attention.head_count_kv": "n_kv_heads",
        "llama.attention.layer_norm_rms_epsilon": "norm_eps",
        "llama.rope.freq_base": "rope_theta",
        "llama.context_length": "max_position_embeddings",
        "tokenizer.ggml.bos_token_id": "bos_token_id",
        "tokenizer.ggml.eos_token_id": "eos_token_id",
        }
        
        for key in key_map:
            if key in config and key_map[key] not in result:
                value = config[key]
                if isinstance(value, mx.array):
                    value = value.item()
                result[key_map[key]] = value
        
        if "general.architecture" in config:
            result["model_type"] = config["general.architecture"]
            
        if "tokenizer.ggml.tokens" in config:
            result["vocab_size"] = len(config["tokenizer.ggml.tokens"])

        if result["vocab_size"] <= 0:
            del result["vocab_size"]
        
        if "head_dim" not in result:
            if "dim" in result and "n_heads" in result:
                result["head_dim"] = result["dim"] // result["n_heads"]

    return result
                