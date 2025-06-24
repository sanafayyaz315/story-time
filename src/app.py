from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
from model import GPTTransformer
from inference import init_tokenizer, load_checkpint, predict

# Setup
max_new_tokens = 2000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
stop_token_id=50256

# init app
stories_app = FastAPI()

# model
tokenizer = init_tokenizer()
model = GPTTransformer(vocab_size=50257, device=device)
model = load_checkpint(model, "../checkpoints", device=device)
model.eval()

print(f"Device: {device}")
print("Model loaded with checkpoints")
print(f"max_new_tokens: {max_new_tokens}")

# Request schema
class InferenceInput(BaseModel):
    input_text: str

# Endpoint
@stories_app.post("/generate")
async def generate(data: InferenceInput):
    input_text = data.input_text

    return StreamingResponse(
        predict(input_text, model, tokenizer, device,  max_new_tokens=2000, stop_token_id=50256),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    print("Application running at http://0.0.0.0:8000")
    uvicorn.run("app:stories_app", host="0.0.0.0", port=8000, reload=True)