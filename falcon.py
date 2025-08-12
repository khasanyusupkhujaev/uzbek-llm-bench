from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, json, os

# Use Falcon model instead of Kimi
model_name = "tiiuae/Falcon3-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Set pad token (Falcon often needs this)
model.config.pad_token_id = tokenizer.eos_token_id

def run_test_on_file(file_path, output_file):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for i, entry in enumerate(data):
        raw_prompt = entry.get("input") or entry.get("prompt") or entry
        if not isinstance(raw_prompt, str):
            raw_prompt = json.dumps(raw_prompt, ensure_ascii=False)

        messages = [{"role": "user", "content": raw_prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200)

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        results.append({
            "input": raw_prompt,
            "output": response.strip()
        })

        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(data)} from {os.path.basename(file_path)}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {output_file}")

# List of JSON test files
test_files = [
    "translate_uz.json",
    "translate_en.json",
    "comprehension.json",
    "generation.json"
]

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

# Run evaluation for each file
for file_name in test_files:
    run_test_on_file(
        os.path.join(file_name),
        os.path.join("results", f"{file_name.replace('.json', '_results.json')}")
    )
