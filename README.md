# mlxllama

**mlxllama** is a lightweight framework for loading and serving Llama-family large language models
using the MLX core library.

## Features

- Load GGUF-format weights with automatic key mapping and optional quantization.
- Support for MLX- and HuggingFace-format models.
- FastAPI-based HTTP API for chat completions (streaming support, temperature, top‑k/p, etc.).

## Repository Structure

- **core/**: Model-loading engine, key mapping, tokenizer wrappers, and LLM abstraction.
- **models/**: Low-level model implementations (transformers, activations, rotary embeddings).
- **server/**: FastAPI application and startup scripts for serving chat completions.
- **utils/**: Helper utilities (e.g. SentencePiece tokenizer builder).

## Quickstart

### Prerequisites

- Python 3.8+
- Dependencies: `mlx`, `sentencepiece`, `fastapi`, `uvicorn`, etc.

### Run the server

```bash
git clone <repo_url>
cd mlxllama
pip install -r requirements.txt
./server/run_server.sh --model /path/to/your-model.gguf
```

The server will listen on port 8000 by default. Send POST requests to:

```http
POST http://0.0.0.0:8000/v1/chat/completions
Content-Type: application/json

{
  "messages": ["Hello, how are you?"],
  "model": "llama",
  "max_tokens": 128,
  "temperature": 0.7
}
```
