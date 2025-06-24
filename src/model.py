import torch
import torch.nn as nn
from torch.nn import functional as F
import os

# Define architecture

# GPT Language Model
class GPTTransformer(nn.Module):
  def __init__(
      self,
      vocab_size,
      n_embd=768,
      n_head=12,
      n_layer=12,
      block_size=1024, 
      dropout=0.1,
      eps=1e-8,
      device="cuda"
  ):
    super().__init__()
    self.vocab_size = vocab_size
    self.n_embd = n_embd
    self.n_head = n_head
    self.n_layer = n_layer
    self.block_size = block_size
    self.dropout = dropout
    self.eps = eps
    self.device = device
    self.use_checkpointing = False
    # nn.embedding creates a weight matrix and extracts the row corrsponding to every value in the input x matrix
    # expanding the model to add more layers after the embedding layer
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # create an embedding lookup table for vocab_size tokens with each token represented by an n_embd size
    self.position_embedding_table = nn.Embedding(block_size, n_embd) # position embedding layer - creates a position embedding for every token position (T num of tokens)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout, eps) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd, eps=eps) # final layer norm just before the lm_head (output layer
    self.lm_head = nn.Linear(n_embd, vocab_size) # final layer of the model called language model head
    self.dropout = nn.Dropout(dropout)
    self.to(device)

  def forward(self, idx, targets=None): # idx is the input token
    idx = idx.to(self.device)
    B, T = idx.shape
    token_emb = self.token_embedding_table(idx) # (B, T, C) This will give us the embedding for every token, hence adding the channel dimension
    pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T, C) to be added to token_emd using broadcasting
    x = token_emb + pos_emb # (B, T, C) # concat the token and pos embeddings. x holds the token identities along with their position info. Not helpful for a bigram model
    x = self.dropout(x) # applying dropout on embeddings
    x = self.blocks(x)
    logits = self.lm_head(x) # (logits) final output (B, T, vocab_size)

    if targets is None:
      loss = None
    else:
      # calculate the loss
      # Pytorch expects (B, C, T) instead of (B, T, C). Instead of reshaping to (B, C, T), we make our logits 2D
      B, T, C = logits.shape
      logits = logits.view(B*T, C) # (32, 65)
      targets = targets.view(B*T) # (32,)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    idx = idx.to(self.device) 
    # Set model to evaluation mode
    self.eval()
    # Use no_grad context manager
    with torch.no_grad():
      # idx is (B, T) array of indices in the current context. B will always be 1 for inference and T will be the num of tokens in the input text
      for _ in range(max_new_tokens):
        # crop idx to get just the last block_size number of tokens else adding pos_emb to token_embd will cause errors since we have pos_embd only for T tokens
        idx_cond = idx[:, -self.block_size:]
        # get predictions
        logits, loss = self(idx_cond) # this will call forward
        # focus only on the last time step to extract the logits for the last token
        logits = logits[:, -1, :] # becomes (B, C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B, C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B,1) num_samples = 1 because we only sample one token at a time. B is 1 so idx_next is a 2D tensor with a single value [[value]]
        # append sampled example to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

    return idx

  def generate_stream(self, idx, max_new_tokens, stop_token_id=None):
    idx = idx.to(self.device) 
    self.eval()
    with torch.no_grad():
      # 1) Yield all tokens from the initial input context first
        for token_id in idx[0].tolist():
            yield token_id

        # 2) Then start generating new tokens and yield them one by one
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Yield the token id (as a Python int)
            token_id = idx_next[0, 0].item()
            
            # Stop if generated the stop token id
            if stop_token_id is not None and token_id == stop_token_id:
                break

            yield token_id

  def enable_checkpointing(self):
      self.use_checkpointing = True

# Blocks: Grouping multihead SELF attention and feedfoward in a block
class Block(nn.Module):
  """Transformer block: communication(attention) followed by computation(linear layers)"""
  def __init__(self, n_embd, n_head, block_size, dropout, eps=1e-8):
    super().__init__()
    self.sa = MultiHeadAttention(n_embd, n_head, block_size, dropout)
    self.ffwd = FeedForward(n_embd, dropout)
    self.ln1 = nn.LayerNorm(n_embd, eps=eps) # layernorm for the multihead-attention layer
    self.ln2 = nn.LayerNorm(n_embd, eps=eps) # layernorm for the feedforward layer

  def forward(self, x):
    # "x + " is adding x for residual connection/skip connection
    # in the original paper layernorm is applied after the self attention layer and the feedfwd layer
    # It is now common to apply it before the self attention layer and the feedfwd layer - so the input to the two sa and ffwd layers will be layernormed input
    # this is called the "pre-norm formulation" and we will be implenting that

    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

# Multi-head Attention
class MultiHeadAttention(nn.Module):
  """multiple heads of self-attention in parallel"""
  def __init__(self, n_embd, n_head, block_size, dropout):
    super().__init__()
    head_size = n_embd // n_head
    self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout) 
            for _ in range(n_head)
            ])
    self.proj = nn.Linear(n_embd, n_embd) # The linear projection layer then mixes (fuses) these stacked features by taking weighted combinations (via the weight matrix) of all head outputs.
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1) # pass the input to each head and concat the output
    out = self.proj(out) # for fusing the information in alll heads and also projecting back into the residual pathway
    out = self.dropout(out) # dropout after the multihead self attention layer
    return out

# Self-Attention Head
class Head(nn.Module):
  """one head of self-attention"""

  def __init__(self, n_embd, head_size, block_size, dropout):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    # Register causal mask (lower triangular) in variable "tril" as non-trainable buffer
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x) # (B, T, head_size)
    q = self.query(x) # (B, T, head_size)
    v = self.value(x) # (B, T, head_size)
    # compute attention wieghts
    wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
    # create mask
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
    # attention scores
    wei = F.softmax(wei, dim=-1) # (B, T, T)
    wei = self.dropout(wei) # # gpt2 small uses dropout on attention scores
    # final attention embeddings
    out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
    return out

# Feedforward class of linear layers
class FeedForward(nn.Module):
  """a simple linear layer followed by a non-linearity"""

  def __init__(self, n_embd, dropout):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd), # the inner layer has a dimensionality (ff_inner_dim) of 4 times the embedding size as per the transformer paper
        nn.GELU(), # GELU activation function
        nn.Linear(4 * n_embd, n_embd), # projection layer going back into the residual pathway -- fixes the shape for x + output of linearlayer
        nn.Dropout(dropout)) # add dropout before passing the connection back into the rsidual pathway

  def forward(self,x):
    return self.net(x)



