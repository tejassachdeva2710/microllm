"""
interactive_chat.py
Real, conversational, intelligent SLM chat running 100% locally on your RTX 3050 Ti.
Uses modern instruction-tuned Small Language Models (SmolLM2-135M-Instruct or 360M-Instruct).

Usage:
  python interactive_chat.py              (Loads 135M-Instruct, ~270 MB, blazingly fast)
  python interactive_chat.py --model 360m (Loads 360M-Instruct, ~720 MB, higher reasoning)
"""
import sys
import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# Fix Windows console cp1252 output encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

MODELS = {
    "135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, witty, and concise AI assistant running locally on the user's laptop. "
    "Keep your answers articulate, accurate, and engaging."
)

def main():
    parser = argparse.ArgumentParser(description="Chat with a local Small Language Model.")
    parser.add_argument("--model", choices=["135m", "360m"], default="135m", help="Model size to load (default: 135m)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference instead of GPU")
    args = parser.parse_args()

    model_id = MODELS[args.model]
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("\n" + "=" * 60)
    print(f"  INITIALIZING LOCAL CONVERSATIONAL SLM: {args.model.upper()}")
    print("=" * 60)
    print(f"Model ID: {model_id}")
    print(f"Device:   {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"Loading weights (first run will download ~270MB to your cache)...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device
    )
    model.eval()

    vram = torch.cuda.memory_allocated() / (1024 ** 2) if device == "cuda" else 0
    print(f"\n[OK] Model Loaded Successfully!")
    if device == "cuda":
        print(f"[OK] VRAM Consumed: {vram:.1f} MB (Leaves 3.5+ GB free!)")
    print("\nCommands:")
    print("  /system <text>  - Change the AI's personality or system prompt")
    print("  /clear          - Reset the conversation history")
    print("  /quit           - Exit the chat")
    print("=" * 60 + "\n")

    system_prompt = DEFAULT_SYSTEM_PROMPT
    history = []

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    while True:
        try:
            user_input = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["/quit", "/exit"]:
            print("Goodbye!")
            break

        if user_input.lower() == "/clear":
            history = []
            print("[History cleared.]\n")
            continue

        if user_input.lower().startswith("/system"):
            new_sys = user_input[7:].strip()
            if new_sys:
                system_prompt = new_sys
                history = []
                print(f"[System prompt updated & history reset: {system_prompt}]\n")
            continue

        # Build conversation message history
        messages = [{"role": "system", "content": system_prompt}]
        for turn in history:
            messages.append(turn)
        messages.append({"role": "user", "content": user_input})

        # Apply chat template
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        print("\nAI > ", end="", flush=True)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                streamer=streamer,
                pad_token_id=tokenizer.eos_token_id
            )

        # Extract generated response text to append to history
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response_text})

        # Keep context bounded (last 8 turns)
        if len(history) > 16:
            history = history[-16:]

        print("\n")

if __name__ == "__main__":
    main()
