"""
chat.py - talk to your model (it is a TEXT COMPLETER until chat fine-tuning is done).
Usage: python chat.py [path_to_checkpoint]
Commands: /temp 0.7   /len 200   /quit
"""
import sys, os, torch, tiktoken
import torch.nn.functional as F
from model import MicroLLM, ModelConfig

default = r"D:\micro-llm\checkpoints_v2\best_model.pt"
if not os.path.exists(default):
    default = r"D:\micro-llm\checkpoints\best_model.pt"
path = sys.argv[1] if len(sys.argv) > 1 else default
device = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load(path, map_location=device, weights_only=False)
cfg = ck["config"]
cfg = ModelConfig(**cfg) if isinstance(cfg, dict) else cfg
model = MicroLLM(cfg).to(device)
model.load_state_dict(ck["model"])
model.eval()
enc = tiktoken.get_encoding("gpt2")
print(f"Loaded {path}\n  {model.count_parameters()/1e6:.2f}M params | trained to step {ck.get('iter_num')} "
      f"| val loss {ck.get('best_val_loss', float('nan')):.4f} | device {device}")
print("Type the START of some text and the model continues it. /temp X, /len N, /quit\n")

temperature, max_new, top_k = 0.8, 150, 40
while True:
    try:
        prompt = input("prompt> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not prompt:
        continue
    if prompt == "/quit":
        break
    if prompt.startswith("/temp"):
        temperature = float(prompt.split()[1]); print(f"temperature = {temperature}"); continue
    if prompt.startswith("/len"):
        max_new = int(prompt.split()[1]); print(f"max tokens = {max_new}"); continue

    idx = torch.tensor([enc.encode(prompt)], device=device)
    new_tokens, printed = [], ""
    print(prompt, end="", flush=True)
    with torch.no_grad():
        for _ in range(max_new):
            logits, _ = model(idx[:, -cfg.max_seq_len:])
            logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            if nxt.item() == enc.eot_token:
                break
            idx = torch.cat([idx, nxt], dim=1)
            new_tokens.append(nxt.item())
            text = enc.decode(new_tokens)
            print(text[len(printed):], end="", flush=True)
            printed = text
    print("\n")
