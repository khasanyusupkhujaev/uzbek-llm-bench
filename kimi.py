from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, json, os

model_name = "moonshotai/Kimi-K2-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.config.pad_token_id = tokenizer.eos_token_id

def run_test_on_file(file_path, output_file):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for i, entry in enumerate(data):
        user_prompt = entry.get("input") or entry.get("prompt") or entry

        messages = [{"role": "user", "content": user_prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200)

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        results.append({
            "input": user_prompt,
            "output": response.strip()
        })

        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(data)} from {os.path.basename(file_path)}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {output_file}")

test_files = [
    "translate_uz.json",
    "translate_en.json",
    "comprehension.json",
    "generation.json"
]

os.makedirs("results", exist_ok=True)

for file_name in test_files:
    run_test_on_file(
        os.path.join(file_name),
        os.path.join("results", f"{file_name.replace('.json', '_results.json')}")
    )
