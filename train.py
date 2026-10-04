"""
train.py:
High-performance training loop for the ~22.5M parameter Micro-LLM.
Optimized for 4GB VRAM (NVIDIA RTX 3050 Laptop) using FP16 mixed precision,
gradient accumulation, memory-mapped data loading, and PyTorch SDPA attention.
"""
import os
import time
import math
import numpy as np
import torch
import tiktoken
from model import MicroLLM, ModelConfig

# ----------------- Hyperparameters & Configuration -----------------
DATA_DIR = r"D:\micro-llm\data"
CHECKPOINT_DIR = r"D:\micro-llm\checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Training Hyperparameters
batch_size = 12            # Micro-batch size (fits comfortably in 4GB VRAM)
block_size = 512           # Context window length
gradient_accumulation_steps = 4  # Effective batch size = 12 * 4 = 48 sequences (~24,576 tokens/step)
max_iters = 5000           # Total training steps
learning_rate = 6e-4       # Peak learning rate
min_lr = 6e-5              # 10% of peak lr
warmup_iters = 200         # Linear warmup steps
lr_decay_iters = 5000      # Cosine decay over entire run
eval_interval = 250        # Evaluate validation loss every N steps
eval_iters = 20            # Number of batches to average for evaluation
save_interval = 500        # Save checkpoint every N steps
sample_interval = 250      # Generate sample text every N steps

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16      # FP16 for RTX 3050 Tensor Cores

print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

# ----------------- Data Loader -----------------
class DataLoader:
    def __init__(self, data_path: str, batch_size: int, block_size: int):
        self.data_path = data_path
        self.batch_size = batch_size
        self.block_size = block_size
        # Memory-map the binary data file for instant access without RAM overhead
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.n_tokens = len(self.data)
        print(f"Loaded {data_path}: {self.n_tokens:,} tokens")

    def get_batch(self):
        ix = np.random.randint(0, self.n_tokens - self.block_size, (self.batch_size,))
        x = torch.stack([torch.from_numpy((self.data[i:i + self.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i + 1:i + 1 + self.block_size]).astype(np.int64)) for i in ix])
        if device == "cuda":
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

# ----------------- Learning Rate Scheduler -----------------
def get_lr(it: int) -> float:
    # 1) Linear warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    # 2) If it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) In between, use cosine decay
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# ----------------- Main Training Script -----------------
def train():
    train_path = os.path.join(DATA_DIR, "train.bin")
    val_path = os.path.join(DATA_DIR, "val.bin")

    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Please run prepare_data.py first!")
        return

    train_loader = DataLoader(train_path, batch_size, block_size)
    val_loader = DataLoader(val_path, batch_size, block_size)

    # Initialize Tokenizer for generation
    enc = tiktoken.get_encoding("gpt2")

    # Initialize Model (~22.5M parameters)
    config = ModelConfig(
        vocab_size=50304,
        dim=288,
        n_layers=8,
        n_heads=6,
        intermediate_size=768,
        max_seq_len=block_size,
        dropout=0.0
    )
    model = MicroLLM(config).to(device)
    num_params = model.count_parameters()
    print(f"\n==========================================")
    print(f"Model Initialized: {num_params / 1e6:.2f} Million Parameters")
    print(f"Target VRAM usage: ~1.5 - 2.5 GB")
    print(f"==========================================\n")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda' and dtype == torch.float16))

    # Validation evaluation helper
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split, loader in [('train', train_loader), ('val', val_loader)]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = loader.get_batch()
                with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=dtype):
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    # Start Training
    model.train()
    best_val_loss = float('inf')
    t0 = time.time()

    for iter_num in range(max_iters):
        # Determine and set learning rate
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluate periodically
        if iter_num % eval_interval == 0:
            losses = estimate_loss()
            vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if device == "cuda" else 0
            print(f"\n>>> [Step {iter_num}/{max_iters}] Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | LR: {lr:.2e} | VRAM: {vram_gb:.2f} GB")

            # Checkpoint best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                checkpoint = {
                    'model': model.state_dict(),
                    'config': config,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "best_model.pt"))
                print(f"    --> Saved best model checkpoint (Val Loss: {best_val_loss:.4f})")

        # Sample generation periodically
        if iter_num > 0 and iter_num % sample_interval == 0:
            prompt = "The secret to the universe is"
            prompt_tokens = enc.encode(prompt)
            x_gen = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
            y_gen = model.generate(x_gen, max_new_tokens=40, temperature=0.8, top_k=40)
            sample_text = enc.decode(y_gen[0].tolist())
            print(f"\n--- [Sample Generation @ Step {iter_num}] ---")
            print(f"Prompt: {prompt}")
            print(f"Output: {sample_text.strip()}")
            print(f"----------------------------------------------\n")

        # Forward & Backward with Gradient Accumulation
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(gradient_accumulation_steps):
            X, Y = train_loader.get_batch()
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=dtype):
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps
            accum_loss += loss.item()
            scaler.scale(loss).backward()

        # Gradient clipping & step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Timing metrics
        if iter_num % 10 == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            tokens_per_sec = (batch_size * block_size * gradient_accumulation_steps * 10) / (dt if dt > 0 else 1)
            if iter_num % 50 == 0:
                print(f"Step {iter_num:4d} | Loss: {accum_loss * gradient_accumulation_steps:.4f} | dt: {dt*100:.1f}ms | tok/sec: {tokens_per_sec:,.0f}")

    print("\nTraining completed successfully!")
    print(f"Final Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train()
