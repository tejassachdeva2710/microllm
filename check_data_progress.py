import os

data_dir = r"D:\micro-llm\data_v2"
if os.path.exists(data_dir):
    for f in os.listdir(data_dir):
        fp = os.path.join(data_dir, f)
        size_mb = os.path.getsize(fp) / (1024 * 1024)
        tokens = os.path.getsize(fp) // 2
        print(f"{f:15}: {size_mb:7.2f} MB ({tokens:,} tokens)")
else:
    print("data_v2 does not exist yet.")
