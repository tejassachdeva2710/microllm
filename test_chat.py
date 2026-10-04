import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
print(f"Downloading/loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")

messages = [
    {"role": "system", "content": "You are a helpful, smart AI assistant running locally."},
    {"role": "user", "content": "Hey! Can you explain what you are and what is quantum physics in 2 simple sentences?"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=120, temperature=0.7)

reply = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("\n" + "=" * 50)
print("TEST CHAT RESPONSE:")
print(reply.strip())
print("=" * 50)
