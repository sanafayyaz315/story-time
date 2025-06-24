## USE THE FOLLOWING LINK TO DOWNLOAD WEIGHTS FROM GOOGLE DRIVE AND PLACE THE FILE IN THE CHECKPOINTS FOLDER
## https://drive.google.com/file/d/1JZ7qipP-0NpZQOA8v7bmsIzM6iCLGvR_/view?usp=sharing


# importing libraries and modules
import sys
import torch
from torch import nn
import tiktoken
import os
from model import GPTTransformer

# defining util functions -- should be moved to utils.py later
# load checkpoint
def load_checkpint(model, checkpoint_dir, device):
  checkpoint_path = os.path.join(checkpoint_dir, "model_step_57302.pt")  # Latest step
  checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
  model.load_state_dict(checkpoint['model_state_dict'])
  # model.eval()  # Set to evaluation modea
  return model

# initialize tiktoken tokenizer
def init_tokenizer():
  tokenizer = tiktoken.get_encoding("gpt2")
  return tokenizer

# tokenize data
def tokenize_data(input_text, tokenizer, device):
  token_ids = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0).to(device)
  return token_ids

# decode data
def decode_data(tokenized_output, tokenizer):
  decoded_input = tokenizer.decode(tokenized_output)
  return decoded_input

def predict(input_text, model, tokenizer, device,  max_new_tokens=2000, stop_token_id=50256):
  # tokenize input_text
  token_ids = tokenizer.encode(input_text)
  # fix shape to add a B dim
  context = torch.tensor(token_ids).unsqueeze(0).to(device)
  for token_id in model.generate_stream(context, max_new_tokens, stop_token_id):
    token = tokenizer.decode([token_id])
    print(token, end="", flush=True)
    # yield f"data: {token}\n\n" ## SSE format but it did not work because of stip() on client.py
    yield token

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    # dir where checkpoint is saved
    checkpoint_dir = "../checkpoints"
    
    # init tokenizer
    tokenizer = init_tokenizer() # vocab size is fixed to 50257
    print(f"vocab size: {tokenizer.n_vocab}")

    # init model with the checkpoint weights
    model = GPTTransformer(vocab_size=50257, device=device)
    model = load_checkpint(model, checkpoint_dir=checkpoint_dir, device=device)
    print(f"Loaded model with checkpoints")

    input_text = "Once upon a time there was a big purple elephant."
    max_new_tokens=2000
    stop_token_id=50256

    # generate
    for token in predict(input_text, model, tokenizer, device, max_new_tokens, stop_token_id):
      print(token, end="", flush=True)



