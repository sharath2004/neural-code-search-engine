from datasets import load_dataset
import json
import os

os.makedirs("data", exist_ok=True)

# Load 1% of the Python subset
dataset = load_dataset("code_search_net", "python", split="train[:1%]")

snippets = []

for item in dataset:
    code = item.get("code")
    doc = item.get("docstring")
    if code and doc:
        snippets.append({"code": code.strip(), "text": doc.strip()})

with open("data/snippets.json", "w") as f:
    json.dump(snippets, f, indent=2)

print(f"✅ Saved {len(snippets)} code-docstring pairs to data/snippets.json")
