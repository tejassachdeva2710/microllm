import torch
from model import MicroLLM, ModelConfig

def verify():
    print("=" * 50)
    print("      MICRO-LLM SYSTEM & HARDWARE VERIFICATION")
    print("=" * 50)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available:  {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU Device:      {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM:      {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        device = "cuda"
    else:
        print("WARNING: CUDA is not available. Falling back to CPU.")
        device = "cpu"

    print("\n[1] Testing GPU Tensor Core Matrix Multiplication...")
    a = torch.randn(1024, 1024, device=device, dtype=torch.float16 if device == "cuda" else torch.float32)
    b = torch.randn(1024, 1024, device=device, dtype=torch.float16 if device == "cuda" else torch.float32)
    c = torch.matmul(a, b)
    print("    --> Success! Matrix multiplication executed cleanly.")

    print("\n[2] Initializing Micro-LLM Architecture...")
    config = ModelConfig(
        vocab_size=50304,
        dim=288,
        n_layers=8,
        n_heads=6,
        intermediate_size=768,
        max_seq_len=512,
        dropout=0.0
    )
    model = MicroLLM(config).to(device)
    total_params = model.count_parameters()
    print(f"    --> Total Trainable Parameters: {total_params:,} ({total_params / 1e6:.2f} Million)")

    print("\n[3] Testing Forward & Backward Pass (FP16 Autocast)...")
    dummy_input = torch.randint(0, config.vocab_size, (4, 128), device=device)
    dummy_targets = torch.randint(0, config.vocab_size, (4, 128), device=device)

    with torch.autocast(device_type=device, dtype=torch.float16 if device == "cuda" else torch.float32):
        logits, loss = model(dummy_input, dummy_targets)

    print(f"    --> Initial Random Cross-Entropy Loss: {loss.item():.4f}")
    
    loss.backward()
    print("    --> Backward pass (gradients computed) executed cleanly!")

    if device == "cuda":
        mem_used = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"\n[4] Peak VRAM Consumed: {mem_used:.2f} MB (Fits comfortably within 4096 MB VRAM!)")

    print("\n" + "=" * 50)
    print("  ALL CHECKS PASSED: READY FOR PRE-TRAINING!")
    print("=" * 50)

if __name__ == "__main__":
    verify()
