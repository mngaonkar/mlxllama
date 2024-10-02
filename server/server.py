import fastapi
import constants
import argparse
import uvicorn

app = fastapi.FastAPI()

@app.get("/")
def read_root():
    return {"message": f"mlxllama version %s" % constants.VERSION}

@app.get("/v1/models")
def read_models():
    return {"models": ["model1", "model2"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)