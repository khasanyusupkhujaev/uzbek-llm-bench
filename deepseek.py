from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, json, os
from nltk.translate.bleu_score import sentence_bleu
import re

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16
).to("cuda")
model.config.pad_token_id = tokenizer.eos_token_id

def clean_translation_output(response):
    """Extract only the translated text from the model's output."""
    # Remove common preambles or extra text
    patterns = [
        r"The translation [^\n]*:\n*\s*",
        r"Translated text:\n*\s*",
        r"Translation:\n*\s*",
        r"^[^\n]+:\n*\s*"  # Remove any introductory line ending with colon
    ]
    cleaned = response
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    # Remove extra whitespace and newlines
    return cleaned.strip()

def run_test_on_file(file_path, output_file):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for i, entry in enumerate(data):
        if file_path.endswith("translate_uz.json") or file_path.endswith("translate_en.json"):
            # Explicitly instruct to return only the translated text
            prompt = f"Translate the following text from {entry['source_lang']} to {entry['target_lang']} and provide only the translated text: {entry['source_text']}"
        elif file_path.endswith("comprehension.json"):
            prompt = f"Based on the following context, answer the question in Uzbek: {entry['context']} {entry['question']}"
        elif file_path.endswith("generation.json"):
            criteria = entry['expected_criteria']
            prompt = f"{entry['prompt']} Write in Uzbek, {criteria['length']}, in a {criteria['style']} style about {criteria['topic']}."
        else:
            prompt = json.dumps(entry, ensure_ascii=False)

        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()

        # Clean the output for translation tasks
        if file_path.endswith("translate_uz.json") or file_path.endswith("translate_en.json"):
            response = clean_translation_output(response)

        # Calculate BLEU score
        expected = entry.get("expected", "")
        bleu_score = sentence_bleu([expected.split()], response.split()) if expected else 0.0

        results.append({
            "input": prompt,
            "output": response,
            "expected": expected,
            "bleu_score": bleu_score
        })

        if (i + 1) % 10 == 0:
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

os.makedirs("results/falcon", exist_ok=True)

for file_name in test_files:
    run_test_on_file(
        os.path.join(file_name),
        os.path.join("results/falcon", f"{file_name.replace('.json', '_results.json')}")
    )
