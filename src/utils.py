# helper funcs
import torch
import math
import os 
import glob
from data import get_batch, load_tokenized_data, tokenize_data
##-------------------------------------##
# Tokenize or Load data
def load_or_tokenize(token_dir, tokenizer, train_data_path, valid_data_path, train_save_path, valid_save_path):
    try:
        # Check if the directory where tokenized data is stored exists
        if not os.path.isdir(token_dir):
            print(f"Directory with tokenized data does not exist")
            raise FileNotFoundError(f"No such directory: {token_dir}")
        # If it exists, check it is not empty
        if not os.listdir(token_dir):
            print(f"Directory with tokenized data is empty")
            raise ValueError(f"Directory is empty: {token_dir}")

        # token_dir exists and isn’t empty - load tokenized data
        print("Loading tokenized data")
        val_data = load_tokenized_data(valid_save_path)
        train_data = load_tokenized_data(train_save_path)
        print("Tokenized data loaded!") 
        return train_data, val_data

    except (FileNotFoundError, ValueError):
        print("Tokenizing data")
        tokenize_data(train_data_path, train_save_path, tokenizer, chunk_size=10000000, allowed_special={"<|endoftext|>"}) # allow special token for boundaries
        tokenize_data(valid_data_path, valid_save_path, tokenizer, chunk_size=10000000, allowed_special={"<|endoftext|>"})
        print(f"Tokenization completed!")
        val_data = load_tokenized_data(valid_save_path)
        train_data = load_tokenized_data(train_save_path)
        print("Tokenized data loaded!") 
        return train_data, val_data

# simply printing the loss after every iteration is noisy because it the loss on a single batch
# a better idea is to view the average of loss instead
# we average the loss for train and test splits after every eval_iters number of batches
@torch.no_grad()
def estimate_loss(model, eval_iters):
    out = {}
    model.eval() # sets the model to eval mode. This is because some layers like dropout, batchnorm etc have different behaviours during train and test/eval time
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            # Ensure autocast is used within evaluation as well, since the model is in FP16
            with torch.autocast(device_type='cuda', dtype=torch.float16): # <-- Added autocast
              logits, loss = model(X, Y) # this calls the model's forward function
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # sets the model to train mode
    return out

def save_checkpoint(model, optimizer, step, val_loss, checkpoint_dir, best=False):
    os.makedirs(checkpoint_dir, exist_ok=True)

    if best:
        save_path = os.path.join(checkpoint_dir, "model_best.pt")
    else:
        save_path = os.path.join(checkpoint_dir, f"model_step_{step}.pt")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'val_loss': val_loss
    }, save_path)
    print(f"Checkpoint saved: {'best' if best else 'step '+str(step)} at {save_path}")

    # Keep only latest 2 non-best checkpoints
    if not best:
        all_ckpts = sorted(
            glob.glob(os.path.join(checkpoint_dir, "model_step_*.pt")),
            key=lambda x: int(x.split("_")[-1].split(".")[0])
        )
        if len(all_ckpts) > 2:
            for ckpt in all_ckpts[:-2]:
                os.remove(ckpt)

def load_latest_checkpoint(model, optimizer, checkpoint_dir, device):
    ckpts = sorted(
        glob.glob(os.path.join(checkpoint_dir, "model_step_*.pt")),
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    if not ckpts:
        print("⚠️ No checkpoint found. Starting training from scratch.")
        return model, optimizer, 0, float('inf')  # No checkpoint found

    latest_ckpt = ckpts[-1]
    checkpoint = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    val_loss = checkpoint.get('val_loss', float('inf'))  # default to inf if not found

    print(f"✅ Resumed from checkpoint: {latest_ckpt}")
    print(f"✅ Resuming from step: {checkpoint['step']}")
    return model, optimizer, checkpoint['step'] + 1, val_loss

def load_best_checkpoint(model, checkpoint_dir, device):
    path = os.path.join(checkpoint_dir, "model_best.pt")
    if not os.path.exists(path):
        raise FileNotFoundError("No best checkpoint found")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Loaded best model.")
    return model

# size of the model
def model_size(model):
    param_size = sum(p.numel() for p in model.parameters())
    print(f"Model size: {param_size/1e6:.2f}M parameters")
    return param_size

# LR scheduler with warmup and decay
def get_lr(step, warmup_steps, max_iters, max_lr=6e-4, min_lr=6e-5):
    """
    Cosine decay with linear warmup.
    - Warmup: Linear increase from 0 to max_lr over `warmup_steps`.
    - Decay: Cosine decay from max_lr to min_lr over remaining steps.
    """
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps  # Linear warmup
    else:
        decay_ratio = (step - warmup_steps) / (max_iters - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * decay_ratio))  # Cosine decay