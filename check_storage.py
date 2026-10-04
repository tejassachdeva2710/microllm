import os
import shutil

total, used, free = shutil.disk_usage("D:\\")
print(f"Total D: Space: {total / (1024**3):.2f} GB")
print(f"Used D: Space:  {used / (1024**3):.2f} GB")
print(f"Free D: Space:  {free / (1024**3):.2f} GB")
print("-" * 45)

for name in os.listdir("D:\\"):
    path = os.path.join("D:\\", name)
    if os.path.isdir(path):
        size = 0
        try:
            for root, _, files in os.walk(path):
                for f in files:
                    try:
                        size += os.path.getsize(os.path.join(root, f))
                    except Exception:
                        pass
        except Exception:
            pass
        if size > 1024**2:
            print(f"{name:30}: {size / (1024**3):.2f} GB")
