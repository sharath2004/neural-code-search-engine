from transformers import RobertaTokenizer, RobertaModel
import torch

class CodeSearchModel:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = RobertaModel.from_pretrained("microsoft/codebert-base")

    def encode(self, texts):
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs.last_hidden_state[:, 0, :]  # CLS token
