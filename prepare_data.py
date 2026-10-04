"""
prepare_data.py:
Streams a curated slice of open-source pre-training data (Dolma sample or FineWeb-Edu / OpenWebText)
and tokenizes it into high-speed memory-mapped binary chunks for training.
"""
import os
import sys
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

DATA_DIR = r"D:\micro-llm\data"
os.makedirs(DATA_DIR, exist_ok=True)

def prepare_slice(num_tokens_target: int = 50_000_000):
    """
    Downloads and tokenizes approximately `num_tokens_target` tokens.
    Uses tiktoken (GPT-2 BPE) for instant, local tokenization.
    """
    print(f"Targeting ~{num_tokens_target / 1e6:.1f}M tokens for pre-training slice...")
    enc = tiktoken.get_encoding("gpt2")
    
    # We use Hugging Face's high-quality open pretraining stream (e.g. FineWeb-Edu sample or Dolma sample)
    print("Connecting to Hugging Face dataset stream...")
    # 'HuggingFaceFW/fineweb-edu' (sample-10BT) or 'allenai/dolma'
    # sample-10BT gives pristine educational web text, perfect for quick convergence of small models
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    
    train_tokens = []
    total_tokens = 0
    
    print("Streaming and tokenizing text...")
    pbar = tqdm(total=num_tokens_target, unit="tok")
    
    for row in dataset:
        text = row.get("text", "")
        if not text:
            continue
        
        # Add <|endoftext|> separator
        tokens = enc.encode(text, allowed_special={"<|endoftext|>"})
        tokens.append(enc.eot_token)
        
        train_tokens.extend(tokens)
        total_tokens += len(tokens)
        pbar.update(len(tokens))
        
        if total_tokens >= num_tokens_target:
            break
            
    pbar.close()
    
    # Split into 95% train and 5% validation
    split_idx = int(len(train_tokens) * 0.95)
    train_arr = np.array(train_tokens[:split_idx], dtype=np.uint16)
    val_arr = np.array(train_tokens[split_idx:], dtype=np.uint16)
    
    train_path = os.path.join(DATA_DIR, "train.bin")
    val_path = os.path.join(DATA_DIR, "val.bin")
    
    train_arr.tofile(train_path)
    val_arr.tofile(val_path)
    
    print(f"Saved {len(train_arr):,} train tokens to: {train_path}")
    print(f"Saved {len(val_arr):,} val tokens to: {val_path}")

if __name__ == "__main__":
    target = 30_000_000  # Default 30M tokens for initial fast sprint
    if len(sys.argv) > 1:
        target = int(sys.argv[1])
    prepare_slice(target)
