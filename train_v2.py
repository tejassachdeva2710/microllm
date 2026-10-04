"""
train_v2.py
  Fresh run:   python -u train_v2.py
  Resume:      python -u train_v2.py --resume
Outputs in D:\\micro-llm\\checkpoints_v2\\ :
  best_model.pt  (lowest val loss, for chat/sharing)
  latest.pt      (model + optimizer, saved every 100 steps, for resuming)
  loss_log.csv   (step, train_loss, val_loss, lr, elapsed_min)
"""
import os, sys, time, math, csv, argparse
from dataclasses import asdict
import numpy as np
import torch
import tiktoken
from model import MicroLLM, ModelConfig

# Fix Windows console cp1252 UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

DATA_DIR = r"D:\micro-llm\data_v2"
CKPT_DIR = r"D:\micro-llm\checkpoints_v2"
os.makedirs(CKPT_DIR, exist_ok=True)

# ---- hyperparameters (tuned for 4 GB RTX 3050 Ti) ----
batch_size = 8
block_size = 512
grad_accum = 6                 # 8*512*6 = 24,576 tokens per step
max_iters = 6000               # ~147M tokens = ~1 pass over a 150M-token dataset
warmup_iters = 200
learning_rate = 6e-4
min_lr = 6e-5
eval_interval = 250
eval_iters = 20
save_latest_every = 100
PROMPTS = ["The secret to the universe is", "Once upon a time,", "The capital of France is"]

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16


class DataLoader:
    def __init__(self, path):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        print(f"Loaded {path}: {len(self.data):,} tokens")

    def get_batch(self):
        ix = np.random.randint(0, len(self.data) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(self.data[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    if it >= max_iters:
        return min_lr
    ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (learning_rate - min_lr)


def main():
    global max_iters
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_iters", type=int, default=max_iters)
    args = ap.parse_args()
    max_iters = args.max_iters

    train_loader = DataLoader(os.path.join(DATA_DIR, "train.bin"))
    val_loader = DataLoader(os.path.join(DATA_DIR, "val.bin"))
    enc = tiktoken.get_encoding("gpt2")

    config = ModelConfig(max_seq_len=block_size)
    model = MicroLLM(config).to(device)
    print(f"Model: {model.count_parameters() / 1e6:.2f}M params | device: {torch.cuda.get_device_name(0)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda")

    start_iter, best_val = 0, float("inf")
    latest_path = os.path.join(CKPT_DIR, "latest.pt")
    log_path = os.path.join(CKPT_DIR, "loss_log.csv")
    if args.resume and os.path.exists(latest_path):
        ck = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_iter = ck["iter_num"] + 1
        best_val = ck["best_val_loss"]
        print(f"RESUMED from step {ck['iter_num']} (best val so far {best_val:.4f})")
    elif not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "train_loss", "val_loss", "lr", "elapsed_min"])

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        out = {}
        for name, loader in (("train", train_loader), ("val", val_loader)):
            losses = []
            for _ in range(eval_iters):
                X, Y = loader.get_batch()
                with torch.autocast("cuda", dtype=dtype):
                    _, loss = model(X, Y)
                losses.append(loss.item())
            out[name] = sum(losses) / len(losses)
        model.train()
        return out

    def save(path, it):
        tmp = path + ".tmp"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "config": asdict(config),
                    "iter_num": it, "best_val_loss": best_val}, tmp)
        os.replace(tmp, path)

    model.train()
    t_start = time.time()
    t0 = time.time()
    for it in range(start_iter, max_iters + 1):
        lr = get_lr(it)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % eval_interval == 0:
            losses = estimate_loss()
            vram = torch.cuda.max_memory_allocated() / 1024 ** 3
            elapsed = (time.time() - t_start) / 60
            print(f"\n>>> [Step {it}/{max_iters}] train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| lr {lr:.2e} | peak VRAM {vram:.2f} GB | {elapsed:.1f} min this session")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([it, f"{losses['train']:.4f}", f"{losses['val']:.4f}", f"{lr:.2e}", f"{elapsed:.1f}"])
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": model.state_dict(), "config": asdict(config),
                            "iter_num": it, "best_val_loss": best_val},
                           os.path.join(CKPT_DIR, "best_model.pt"))
                print(f"    saved best_model.pt (val {best_val:.4f})")
            if it > 0:
                for p in PROMPTS:
                    x = torch.tensor([enc.encode(p)], device=device)
                    y = model.generate(x, max_new_tokens=40, temperature=0.8, top_k=40)
                    decoded = enc.decode(y[0].tolist(), errors='replace')
                    try:
                        print(f"    [{p}] -> {decoded!r}")
                    except Exception:
                        print(f"    [{p}] -> {decoded.encode('ascii', errors='replace').decode('ascii')!r}")
                model.train()

        if it == max_iters:
            break

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(grad_accum):
            X, Y = train_loader.get_batch()
            with torch.autocast("cuda", dtype=dtype):
                _, loss = model(X, Y)
                loss = loss / grad_accum
            step_loss += loss.item()
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if it % 50 == 0 and it > start_iter:
            dt = time.time() - t0
            t0 = time.time()
            tps = 50 * batch_size * block_size * grad_accum / dt
            print(f"step {it:5d} | loss {step_loss:.4f} | {dt / 50:.2f} s/step | {tps:,.0f} tok/s")

        if it % save_latest_every == 0 and it > start_iter:
            save(latest_path, it)

    save(latest_path, max_iters)
    print(f"\nDONE. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
