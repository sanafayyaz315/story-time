import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from datasets import load_dataset
from transformers import GPT2LMHeadModel, TrainingArguments, Trainer
import tiktoken
import os
import math
import glob
from data import tokenize_data, load_tokenized_data, get_batch
from model import GPTTransformer
from utils import load_or_tokenize, estimate_loss, save_checkpoint, load_latest_checkpoint, get_lr

##-------------------------------SETUP VARIABLES-------------------------------------##

# gpt2-small model architecture specs
n_embd = 768 # embedding dims
n_head = 12 # num of attention heads in each transformer block
n_layer = 12 # num of blocks
vocab_size = 50257
block_size = 1024 # context length
ff_inner_dim = 3072 # inner dim of feedforward layer
ln_eps = 1e-5 # epsilon for layer norm
dropout = 0.1
activation = "gelu"
print(vocab_size)

# gpt2-small training
# Batch Size & Gradient Accumulation
# in case of OOM, reduce batch_size (e.g., to 4) and increase accumulation_steps
batch_size = 8                  # Physical batch size (per GPU)
block_size = 1024               # context length
effective_batch_size = 32       # Logical batch size (after accumulation)
accumulation_steps = 4          # effective_batch_size / batch_size

# Learning Rate & Schedule
learning_rate = 6e-4            # GPT-2's max LR (original: 6e-4)
warmup_steps = 1500             # ~3% of max_iters (or 1-2% if training longer)
lr_schedule = "cosine"          # With warmup (see optimizer setup below)

# Optimizer (AdamW)
betas = (0.9, 0.95)            # GPT-3 style (more stable than 0.999)
weight_decay = 0.1              # GPT-2 default (better regularization than 0.01)
eps = 1e-8                      # Epsilon for numerical stability
grad_clip = 1.0                 # Clip gradients to prevent explosions

# Evaluation & Checkpoints
eval_interval = 500             # Check val loss every 500 steps
eval_iters = 200                # Use 200 batches to estimate val loss
checkpoint_interval = 2000     # Save every 2000 steps
# Create Checkpoint Directory
checkpoint_dir = "../checkpoints/tiny_stories_ckpts"
os.makedirs(checkpoint_dir, exist_ok=True)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# tokenize train and val splits and save .bin files
token_dir = "../data/tokenized/tiny_stories"
train_data_path = '../data/TinyStories-train.txt'
valid_data_path = '../data/TinyStories-valid.txt'
train_save_path = '../data/tokenized/tiny_stories/train/train'
valid_save_path = '../data/tokenized/tiny_stories/valid/valid'
# tokenization
chunk_tokens = []
chunk_size = 10000000  # tokens per chunk, tune based on your RAM
chunk_idx = 0
special_token = "<|endoftext|>"

##-------------------------------TRAIN FUNCTION-------------------------------------##
def train(model, train_data, val_data, optimizer, device, checkpoint_dir, checkpoint_interval,
          batch_size=8, block_size=1024, max_iters=57303, warmup_steps=1500,
          eval_interval=500, eval_iters=200):

    model.to(device)
    model.train()
    scaler = torch.cuda.amp.GradScaler()

    # Try resuming from latest checkpoint
    model, optimizer, start_iter, best_val_loss = load_latest_checkpoint(model, optimizer, checkpoint_dir, device)
    if start_iter == 0:
      print("Starting training from scratch.")
    else:
        print(f"Resuming training from step: {start_iter}")
    # Initialize current_val_loss to a placeholder value
    current_val_loss = float('inf') # Initialize to a high value

    for step in range(start_iter, max_iters):
        # Learning rate scheduling
        current_lr = get_lr(step, warmup_steps, max_iters, max_lr=6e-4, min_lr=6e-5)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Forward + loss
        x, y = get_batch('train')
        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits, loss = model(x, y)

        # Backward + gradient accumulation
        scaler.scale(loss).backward()
        if (step + 1) % 4 == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % 100 == 0:
            print(f"Step {step}: train_loss={loss.item():.4f}, lr={current_lr:.2e}")

        # Validation + checkpointing
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(model=model, eval_iters=eval_iters)
            current_val_loss = losses['val']
            print(f"\nStep {step}/{max_iters}:")
            print(f"  Train loss: {losses['train']:.4f} | Val loss: {current_val_loss:.4f}")
            print(f"  Learning rate: {current_lr:.2e}")

            is_best = current_val_loss < best_val_loss
            if is_best:
                best_val_loss = current_val_loss
                print(f"🌟 New best validation loss: {best_val_loss:.4f}")
                save_checkpoint(model, optimizer, step, best_val_loss, checkpoint_dir, best=True)

        # Periodic checkpointing (non-best)
        if step % checkpoint_interval == 0 or step == max_iters - 1:
            save_checkpoint(model, optimizer, step, current_val_loss, checkpoint_dir, best=False)
            print(f"Saved checkpoint at step {step}")

    return model

if __name__ == "__main__":

    ##-------------------------------TOKENIZATION-------------------------------------##
    # tokenize data
    # Initialize tiktoken as the tokenizer for gpt2 small
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"Tokenizer vocabulary size: {tokenizer.n_vocab}")
    print(f"token id for special token:{tokenizer.eot_token}")
    print(f"special token: {tokenizer.decode([tokenizer.eot_token])}")

    # tokenize or load tokenized data
    train_data, val_data = load_or_tokenize(token_dir, tokenizer, train_data_path, valid_data_path, train_save_path, valid_save_path)
    print(f"Length of train data: {len(train_data)}")
    print(f" Length of validation data: {len(train_data)}")

    ##-------------------------------MODEL INIT-------------------------------------##
    # model initialization
    print(f"Using device: {device}")
    model = GPTTransformer(vocab_size=50257, device=device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params/1e6:.2f}M parameters")

    # initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=6e-4,                # Peak LR (matches GPT-2 paper)
        betas=(0.9, 0.95),      # GPT-3 style (more stable than 0.999)
        weight_decay=0.1,        # Stronger regularization
        eps=1e-8
    )
    # Enable gradient checkpointing if using large models (optional)
    if n_layer > 12:  # Only needed for very deep models
        model.enable_checkpointing()

    ##-------------------------------TRAIN-------------------------------------##
    # Training Duration
    train_data_size = len(train_data)
    val_data_size = len(val_data)
    num_epochs = 1
    max_iters = int(num_epochs * (train_data_size // (batch_size * block_size)))
    print(f"max number of iterations: {max_iters}")

    # start training
    train(
        model=model,
        train_data=train_data,  # Your preprocessed training tokens
        val_data=val_data,      # Your preprocessed validation tokens
        optimizer=optimizer,
        device=device,          # 'cuda' or 'cpu'
        checkpoint_dir=checkpoint_dir,
        batch_size=batch_size,
        block_size=block_size,
        max_iters=max_iters,        # Total steps (1 epoch for TinyStories)
        warmup_steps=warmup_steps,      # ~3% of max_iters
        eval_interval=eval_interval,      # Evaluate every 500 steps
        checkpoint_interval=checkpoint_interval, # Save checkpoint every 1000 steps
        eval_iters=eval_iters          # Number of batches for validation loss
    )





