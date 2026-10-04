# AGENT RUNBOOK: Micro-LLM Project (for Gemini or any follow-up agent)

You are continuing a project for Tejas (17, India). Read this whole file before you run anything.
Follow the steps **in order**. Don't skip the verification checks. Don't improvise new architecture or new plans.

---

## 0. Context (what already exists)

- Project folder: `D:\micro-llm` (SanDisk external SSD, ~200 GB free).
- Python venv: `D:\micro-llm\venv`. **Always** use `D:\micro-llm\venv\Scripts\python.exe` (never bare `python`, which is a CPU-only install).
- GPU: RTX 3050 Ti Laptop, **4 GB VRAM**, runs hot (hit 87°C under load).
- Torch: 2.6.0+cu124 (CUDA works, verified).
- Model: `model.py`, a LLaMA/OLMo-style transformer (RoPE, RMSNorm, SwiGLU, tied embeddings), **22,455,072 params**. **Don't edit model.py.**
- **v1 run (old):** `train.py` on `data\` (5M tokens of FineWeb-Edu). Checkpoint: `checkpoints\best_model.pt`. It was overfitting (5M tokens is too few). It's kept only as a backup.
- **v2 files (new, written by Claude, use these):**
  - `prepare_data_v2.py`: streams real **Dolma v1.7** shards (falls back to FineWeb-Edu automatically) into `data_v2\`, and writes `data_v2\DATASET.txt` recording which dataset was ACTUALLY used.
  - `train_v2.py`: batch 8 × accum 6, correct loss printing, CSV log, **pause/resume** (`--resume`).
  - `chat.py`: interactive prompt completion.
  - `plot_loss.py`: makes `loss_curve.png`.

## 1. Hard rules (break these and you damage the project)

1. **Never invent numbers.** Every loss value, speed, VRAM figure or step count you tell the user must be copied from real log output. If you haven't seen it in a log, don't say it.
2. **Dataset honesty.** Only call it "Dolma" if `data_v2\DATASET.txt` says `allenai/dolma`. Otherwise it's "FineWeb-Edu". This matters because the user will post it publicly and use it in grant applications.
3. **PowerShell quoting:** `powershell -Command "..."` containing `$variables` breaks (the `$` gets eaten). For anything more complex than one simple command, **write a small .py file and run it with the venv python** instead.
4. Don't delete anything outside `D:\micro-llm`. Don't delete `checkpoints\` (v1) or `checkpoints_v2\`.
5. Don't run two GPU jobs at once (4 GB VRAM). Don't start training while the user says they're gaming.
6. Don't poll task status in a tight loop. Check every few minutes at most.
7. Don't over-promise. A 22M model writes fluent but shallow, often wrong text. It won't chat like ChatGPT. Say so if asked.

---

## 2. Step-by-step: TONIGHT

### Step 2.1: Stop the v1 run (after it passes step 1000)
- Check the running v1 training task's log (a background task launched with `train.py`; use `manage_task list` to find it).
- Once the log shows `>>> [Step 1000/5000]` and `Saved best model checkpoint`, **kill that task** (`manage_task kill`).
- If it's already past 1000, just kill it. The best checkpoint is saved automatically.

### Step 2.2: Install matplotlib
```
D:\micro-llm\venv\Scripts\python.exe -m pip install matplotlib
```

### Step 2.3: Quick test of v1 model with chat.py (optional, 1 min, fun for the user)
```
D:\micro-llm\venv\Scripts\python.exe D:\micro-llm\chat.py D:\micro-llm\checkpoints\best_model.pt
```
chat.py is interactive, so tell the USER to run it themselves in their own terminal (don't run it as a background task). Explain that it continues text; it doesn't answer questions yet.

### Step 2.4: Build the 150M-token dataset
Run in background (takes ~5–20 min depending on internet):
```
D:\micro-llm\venv\Scripts\python.exe -u D:\micro-llm\prepare_data_v2.py 150000000 dolma
```
**Verify when done:**
- Output ends with a block showing `dataset:`, `train_tokens:`, `val_tokens:`.
- `train_tokens` ≈ 148,500,000 and `val_tokens` ≈ 1,500,000.
- `D:\micro-llm\data_v2\train.bin` is ~297 MB (2 bytes per token).
- Read `data_v2\DATASET.txt` and **tell the user which dataset was used**.

**If it fails:**
- Dolma errors are handled automatically (it falls back to FineWeb). If FineWeb ALSO fails, it's a network/HF issue: retry once, then report the exact error to the user.
- If it's extremely slow (<50k tok/s for 10+ minutes), Dolma shard downloads are slow. You may kill it and rerun with `fineweb` as the 2nd argument instead of `dolma`. Tell the user.

### Step 2.5: Check GPU temperature before training
```
nvidia-smi --query-gpu=temperature.gpu,memory.used --format=csv
```
Tell the user to put the laptop on a hard surface or cooling pad with vents clear. If idle temp is >75°C, wait a few minutes.

### Step 2.6: Start v2 training (background, ~4–5 hours)
```
D:\micro-llm\venv\Scripts\python.exe -u D:\micro-llm\train_v2.py
```
Run with Cwd `D:\micro-llm` (model.py import needs it).

**Verify in the first ~3 minutes:**
- Log shows `Model: 22.46M params`.
- `>>> [Step 0/6000] train ~10.8 | val ~10.8` (random-guess baseline, correct).
- `step 50 | loss ...` lines appear. Loss printed there should be around 9–10 early, **NOT ~40** (if ~40, the old buggy train.py is running by mistake).
- Record the `s/step` value. Expected roughly 1.5–3 s/step. Estimated total time = s/step × 6000.

**Then check `nvidia-smi` once:**
- If memory.used ≥ 4000 MiB and s/step > 3.5: VRAM is spilling. Kill it, edit `train_v2.py` → `batch_size = 6`, `grad_accum = 8`, delete `checkpoints_v2\` contents, restart.
- If temperature ≥ 90°C sustained: warn the user. Suggest improving airflow. Don't ignore it.

### Step 2.7: What healthy progress looks like
| Step | Expected val loss (approx.) |
|---|---|
| 0 | ~10.8 |
| 500 | ~6.0–6.5 |
| 1000 | ~5.0–5.5 |
| 3000 | ~4.0–4.5 |
| 6000 | ~3.6–4.2 |

These are rough expectations, not promises. Report the real numbers.
**Warning sign:** val loss going UP for 2+ evals in a row while train loss keeps dropping means overfitting. Report it.
**Warning sign:** loss shows `nan`. Kill it and report to the user.

---

## 3. Pause / resume (when the user wants to game)

- **Pause:** kill the training task. Progress is saved to `checkpoints_v2\latest.pt` every 100 steps (at most ~5 min of work lost).
- **Resume:**
```
D:\micro-llm\venv\Scripts\python.exe -u D:\micro-llm\train_v2.py --resume
```
- Verify the log says `RESUMED from step N`.
- **Never** run without `--resume` if `checkpoints_v2\latest.pt` exists, unless the user explicitly wants to start over (it would restart from step 0).

---

## 4. After training finishes (`DONE. Best val loss: X` appears)

1. Make the graph:
   `D:\micro-llm\venv\Scripts\python.exe D:\micro-llm\plot_loss.py` → creates `D:\micro-llm\loss_curve.png`.
2. Tell the user to run `chat.py` (no argument; it auto-picks `checkpoints_v2\best_model.pt`) and try prompts like:
   - `The capital of France is`
   - `Once upon a time, a young engineer`
   - `Photosynthesis is the process`
3. Collect for the user's posts (all copied from REAL logs):
   - final best val loss, total steps, tokens trained (steps × 24,576), dataset name from DATASET.txt, average s/step and tok/s, peak VRAM, 3 good sample outputs (cherry-picking is fine if disclosed).

---

## 5. Next phase (Day 2: proof of work). Do only when the user asks.

### 5.1 README.md for GitHub
Create `D:\micro-llm\README.md` with these sections:
- One-line summary: "A 22.5M-parameter LLaMA/OLMo-style transformer trained from scratch on a single RTX 3050 Ti laptop (4 GB VRAM)."
- Architecture table (dim 288, 8 layers, 6 heads, SwiGLU 768, RoPE, RMSNorm, tied embeddings, context 512, GPT-2 tokenizer).
- Data (exact text from DATASET.txt).
- Training setup (batch 8 × accum 6 × 512 = 24,576 tok/step, AdamW lr 6e-4 cosine → 6e-5, warmup 200, fp16 AMP).
- Results: loss_curve.png + real final numbers + sample outputs.
- Limitations (honest: small model, shallow knowledge, makes things up).
- How to run (venv, prepare_data_v2, train_v2, chat).

### 5.2 .gitignore (CRITICAL: keep the repo small)
```
venv/
data/
data_v2/
checkpoints/
checkpoints_v2/
*.pt
*.bin
__pycache__/
```
Then: `git init`, add, commit. The user creates the GitHub repo and pushes (needs their login). Never push without the user's OK.

### 5.3 Hugging Face upload
Only `checkpoints_v2\best_model.pt` (~90 MB) + `model.py` + README. The user must log in (`huggingface-cli login`) themselves. Never ask for or store their tokens in files.

---

## 6. Things to NOT do
- Don't change hyperparameters "to make it better" without telling the user why.
- Don't switch to a bigger model. 22.5M is the right size for 4 GB and this timeline.
- Don't start the NPC/Godot phase until v2 training is done and the user asks.
- Don't write grant applications containing numbers that aren't in the logs.

## 7. How to report to the user
Keep it short. Format:
- **Status:** (running / done / error) at step X of 6000
- **Real numbers:** train/val loss, s/step, temp, from the log
- **ETA:** remaining steps × s/step
- **Next action:** what you'll do or what they need to do
