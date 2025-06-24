import torch
import torch.nn as nn
from datasets import load_dataset
import tiktoken
import os
import glob
##----------------------------------------------##

# init variables
chunk_tokens = []
chunk_size = 10000000  # tokens per chunk, tune based on your RAM
chunk_idx = 0
special_token = "<|endoftext|>"

# tokenize train and val splits and save .bin files
train_data_path = '../data/TinyStories-train.txt'
valid_data_path = '../data/TinyStories-valid.txt'
train_save_path = '../data/tokenized/tiny_stories/train/train'
valid_save_path = '../data/tokenized/tiny_stories/valid/valid'

# Tokenize chunks of data and store in separate files for memory efficiency
def tokenize_data(file_path, output_path, tokenizer, chunk_size=10000000, allowed_special={}):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # ensure output directory exists

    chunk_tokens = []
    chunk_idx = 0

    with open(file_path, 'r') as f:
        for line in f:
            tokens = tokenizer.encode(line, allowed_special=allowed_special)
            chunk_tokens.extend(tokens)

            if len(chunk_tokens) >= chunk_size:
                save_chunk(chunk_tokens, chunk_idx, output_path)
                chunk_tokens = []
                chunk_idx += 1

        # Save any leftover tokens
    if chunk_tokens:
        save_chunk(chunk_tokens, chunk_idx, output_path)

# To save tokenized data into chunks instead of a singke file
def save_chunk(tokens, idx, save_file_path):
    tensor = torch.tensor(tokens, dtype=torch.long)
    path = f'{save_file_path}_{idx}.bin'
    torch.save(tensor, path)
    print(f'Saved chunk {idx} with {len(tokens)} tokens at location {path}')

# Load all the tokenized chunks and concat in a tensor
def load_tokenized_data(file_path):
    chunks = []
    i = 0
    while True:
        try:
            chunk = torch.load(f'{file_path}_{i}.bin')
            chunks.append(chunk)
            print(f'Loaded chunk {i} with {len(chunk)} tokens')
            i += 1
        except FileNotFoundError:
            print(f'No more chunks found after index {i-1}, loading complete.')
            break

    return torch.cat(chunks) if chunks else torch.tensor([], dtype=torch.long)


##-----------------------------DATA LOADER---------------------------------##
def get_batch(train_data, val_data, split, block_size, batch_size, device):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) # get batch_size (B) random indices, shape (B,)
    x = torch.stack([data[i:i+block_size] for i in ix]) # shape will be (B,T)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # shape will be (B ,T)
    x, y = x.to(device), y.to(device) # move the data to device (this is important in case device == cuda. we need to move the data to the gpu after loading)
    return x, y

if __name__ == "__main__":
    # Initialize tiktoken as the tokenizer for gpt2 small
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"Tokenizer vocabulary size: {tokenizer.n_vocab}")
    print(f"token id for special token:{tokenizer.eot_token}")
    print(f"special token: {tokenizer.decode([tokenizer.eot_token])}")
    # tokenize train data
    tokenize_data(train_data_path, train_save_path, tokenizer, chunk_size=chunk_size, allowed_special={special_token}) # allow special token for boundaries
    # tokenize val data
    tokenize_data(valid_data_path, valid_save_path, tokenizer, chunk_size=chunk_size, allowed_special={special_token})

