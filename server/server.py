import fastapi
import constants
import argparse
import uvicorn
import os
import sys
from typing import List
import pydantic
import logging

logger = logging.getLogger(__name__)

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import loader

class ChatCompletionRequest(pydantic.BaseModel):
    """Chat completion request."""
    messages: List[str]
    # logprobs: bool = False
    model: str = "llama"
    max_tokens: int = 4096
    temperature: float = 0.7

app = fastapi.FastAPI()

@app.get("/")
def read_root():
    return {"message": f"mlxllama version %s" % constants.VERSION}

@app.get("/v1/models")
def read_models():
    return {"models": ["model1", "model2"]}

@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    args = {
        # "logprobs": request.logprobs,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature
    }

    prompt = " ".join(request.messages)
    logger.info(f"Prompt: {prompt}")
    content, metadata = app.model.completion(prompt, **args)
    return content

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="directory containing the model")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    # Load model
    model = loader.load(args.model)
    app.model = model

    uvicorn.run(app, host=args.host, port=args.port)