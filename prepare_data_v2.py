"""
prepare_data_v2.py
Usage:  python prepare_data_v2.py [num_tokens] [dolma|fineweb]
Default: 150,000,000 tokens, source = dolma (auto-falls back to fineweb if Dolma fails).

Writes to D:\\micro-llm\\data_v2\\ : train.bin, val.bin (uint16 GPT-2 tokens), DATASET.txt
Tokens are flushed to disk in chunks, so RAM use stays low.
"""
import os, sys, json, gzip, random, datetime
import numpy as np
import tiktoken
from tqdm import tqdm

OUT_DIR = r"D:\micro-llm\data_v2"
VAL_FRACTION = 0.01          # 1% of tokens go to validation
FLUSH_EVERY = 2_000_000      # tokens buffered before writing to disk


def dolma_text_stream():
    """Streams documents from real Dolma shards (no HF loading script needed)."""
    import requests
    from huggingface_hub import hf_hub_download
    url_file = hf_hub_download(repo_id="allenai/dolma", filename="urls/v1_7.txt", repo_type="dataset")
    with open(url_file, "r", encoding="utf-8") as f:
        urls = [u.strip() for u in f if u.strip()]
    random.Random(42).shuffle(urls)   # mix of sources (web, wiki, books, papers, code...)
    print(f"Dolma v1.7: {len(urls)} shard URLs found. Streaming shards in shuffled order.")
    for url in urls:
        print(f"  -> shard: {url}")
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                r.raw.decode_content = False
                with gzip.GzipFile(fileobj=r.raw) as gz:
                    for line in gz:
                        try:
                            text = json.loads(line).get("text", "")
                        except Exception:
                            continue
                        if text:
                            yield text
        except Exception as e:
            print(f"     shard failed ({e}); skipping to next shard")
            continue


def fineweb_text_stream():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for row in ds:
        text = row.get("text", "")
        if text:
            yield text


def main(target_tokens: int, source: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    val_target = int(target_tokens * VAL_FRACTION)

    used_source = source
    if source == "dolma":
        try:
            stream = dolma_text_stream()
            first = next(stream)                    # prove Dolma actually works
            def chained(first=first, stream=stream):
                yield first
                yield from stream
            stream = chained()
            dataset_name = "allenai/dolma v1.7 (raw shards, shuffled, seed 42)"
        except Exception as e:
            print(f"!! Dolma failed: {e}\n!! Falling back to FineWeb-Edu.")
            used_source = "fineweb"
    if used_source == "fineweb":
        stream = fineweb_text_stream()
        dataset_name = "HuggingFaceFW/fineweb-edu (sample-10BT)"

    train_path = os.path.join(OUT_DIR, "train.bin")
    val_path = os.path.join(OUT_DIR, "val.bin")
    n_val = n_train = 0
    buf = []

    def flush():
        nonlocal buf, n_val, n_train
        if not buf:
            return
        arr = np.array(buf, dtype=np.uint16)
        buf = []
        need_val = max(0, val_target - n_val)
        if need_val > 0:
            v = arr[:need_val]
            with open(val_path, "ab") as f:
                v.tofile(f)
            n_val += len(v)
            arr = arr[need_val:]
        if len(arr):
            with open(train_path, "ab") as f:
                arr.tofile(f)
            n_train += len(arr)

    # start fresh
    for p in (train_path, val_path):
        if os.path.exists(p):
            os.remove(p)

    pbar = tqdm(total=target_tokens, unit="tok")
    total = 0
    for text in stream:
        toks = enc.encode_ordinary(text)
        toks.append(enc.eot_token)
        buf.extend(toks)
        total += len(toks)
        pbar.update(len(toks))
        if len(buf) >= FLUSH_EVERY:
            flush()
        if total >= target_tokens:
            break
    flush()
    pbar.close()

    info = (
        f"dataset: {dataset_name}\n"
        f"tokenizer: tiktoken gpt2 (50257 vocab)\n"
        f"train_tokens: {n_train}\n"
        f"val_tokens: {n_val}\n"
        f"created: {datetime.datetime.now().isoformat(timespec='seconds')}\n"
    )
    with open(os.path.join(OUT_DIR, "DATASET.txt"), "w", encoding="utf-8") as f:
        f.write(info)
    print("\n" + info)


if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 150_000_000
    src = sys.argv[2] if len(sys.argv) > 2 else "dolma"
    main(target, src)
