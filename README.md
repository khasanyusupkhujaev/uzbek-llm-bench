**Uzbek LLM Benchmarking Repository**

This repository contains the code, datasets, and results for the research paper "Benchmarking Pre-Trained Open-Source Large Language Models for Uzbek: Evaluating Performance in a Low-Resource Setting Across Translation, Comprehension, and Generation" by K. Yusupkhujaev. The study evaluates seven small-scale (0.6B–2B parameters) pre-trained open-source LLMs on Uzbek language tasks: translation (Uzbek↔English), comprehension, and generation. The repository supports reproducibility and provides resources for researchers and practitioners working on Uzbek NLP under constrained compute budgets.
Please cite the paper if you use this repository:

K. Yusupkhujaev, "Benchmarking Pre-Trained Open-Source Large Language Models for Uzbek: Evaluating Performance in a Low-Resource Setting Across Translation, Comprehension, and Generation," 2025. [Online]. Available: Soon will be available online

**Repository Structure**

dataset/: Contains JSON files with the evaluation dataset (180 examples) covering translation, comprehension, and generation tasks.
translate_uz.json: 50 Uzbek-to-English translation sentence pairs.
translate_en.json: 50 English-to-Uzbek translation sentence pairs.
comprehension.json: 30 passages with 1–3 questions each for reading comprehension.
generation.json: 50 prompts for open-ended generation (e.g., narratives, summaries).


models/: Python scripts for testing the seven LLMs using the Hugging Face Transformers library.
deepseek.py: Evaluates DeepSeek-R1-Distill-Qwen-1.5B.
falcon.py: Evaluates Falcon3-1B-Instruct.
gemma.py: Evaluates Gemma-2-2B-it.
glm.py: Evaluates GLM-Edge-1.5B-Chat.
kimi.py: Evaluates Kimi-K2-Instruct.
llama.py: Evaluates Llama-3.2-1B-Instruct.
qwen.py: Evaluates Qwen3-0.6B.


results/: Output files from model evaluations, organized by model.
Subdirectories: deepseek/, falcon/, gemma/, glm/, kimi/, llama/, qwen/.
Each subdirectory contains:
comprehension_results.json: Model outputs for comprehension tasks.
generation_results.json: Model outputs for generation tasks.
translation_results_uz.csv: Uzbek-to-English translation results.
translation_results_en.csv: English-to-Uzbek translation results.
figure_1.png: Plot of per-example BLEU scores for translations.





**Dataset Description**
The dataset (dataset/) comprises 180 examples curated to reflect real-world Uzbek usage across domains: formal, news, technical, conversational, and jargon. All examples were manually verified by a native Uzbek speaker. Details:

**Translation**: 100 sentence pairs (50 Uz→En, 50 En→Uz), ranging from simple (e.g., "Men uyda o‘tiribman" → "I am sitting at home") to complex technical/jargon sentences.
**Comprehension**: 30 passages (200–400 words) with 1–3 extractive/inferential questions each, adapted from sources like Wikipedia and SQuAD.
**Generation**: 50 prompts for open-ended tasks (e.g., writing a short story about Uzbekistan’s Independence Day).

See Section 3.2 of the paper for full dataset construction details.

**Model Testing**
The scripts in models/ evaluate the seven LLMs using zero-shot prompting on a RunPod A40 GPU with the Hugging Face Transformers library (PyTorch, batch size 1, max tokens 512, temperature 0.7). Each script (e.g., kimi.py) loads a pre-trained model, processes the dataset files, and saves results to the corresponding results/ subdirectory.

**Prerequisites:**
Python 3.8+
Libraries: transformers, torch, json, os
Hugging Face account and token for model access (set as environment variable HF_TOKEN)
GPU (e.g., NVIDIA A40) recommended for inference

**Setup**

Clone the repository:git clone https://github.com/khasanyusupkhujaev/uzbek-llm-bench
cd uzbek-llm-bench


Install dependencies: pip install transformers torch nltk accelerate sacremoses protobuf safetensors


Set Hugging Face token:export HF_TOKEN="your_hugging_face_token"



**Running the Code**
Each script in models/ processes all dataset files and saves results to results/<model_name>/. Example for Kimi-K2-Instruct:
python models/kimi.py

This generates:

results/kimi/translate_uz_results.json
results/kimi/translate_en_results.json
results/kimi/comprehension_results.json
results/kimi/generation_results.json

To run all models, execute each script:
for script in models/*.py; do python $script; done

**Notes**

Ensure sufficient GPU memory (A40 or equivalent) for models like Gemma-2-2B-it.
Scripts assume dataset/ files are in the root directory. Adjust paths if needed.
Results are saved as JSON/CSV for analysis; BLEU score plots require additional processing (see paper for details).

**Results**
The results/ directory contains model outputs and evaluation metrics:

JSON Files: Raw model outputs for comprehension and generation tasks.
CSV Files: Translation results with input-output pairs.
PNG Files: BLEU score distribution plots (e.g., figure_5.png for Kimi-K2-Instruct, showing strong performance).

Key finding: Kimi-K2-Instruct outperformed other models, achieving an average BLEU of 0.54 and robust comprehension/generation performance. See Section 4 of the paper for detailed results.

**Reproducing Figures**
To reproduce BLEU score plots (Figures 1–7 in the paper):

Use translation_results_uz.csv and translation_results_en.csv for each model.
Compute per-example BLEU scores using SacreBLEU (see paper, Section 3.3).
Plot sorted BLEU scores (e.g., using Python with matplotlib).

Example (requires sacrebleu and matplotlib):
pip install sacrebleu matplotlib

Then, adapt your own plotting script based on the paper’s methodology.

**License**
This repository is licensed under the MIT License. See LICENSE for details.

**Contact**
For questions, contact Khasan Yusupkhujaev at khasanya99@gmail.com or open an issue on GitHub.
