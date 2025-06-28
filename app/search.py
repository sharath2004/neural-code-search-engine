import json
import torch
import faiss
from app.model import CodeSearchModel
from sklearn.preprocessing import normalize

class CodeSearchEngine:
    def __init__(self, data_path="data/snippets.json"):
        self.model = CodeSearchModel()
        self.codes = []
        self.texts = []
        self.index = None
        self._load_data(data_path)
        self._build_index()

    def _load_data(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        self.codes = [item["code"] for item in data]
        self.texts = [item["text"] for item in data]

    def _build_index(self):
        embeddings = self.model.encode(self.texts).detach().numpy()
        embeddings = normalize(embeddings)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def search(self, query, top_k=3):
        query_vec = self.model.encode([query]).detach().numpy()
        query_vec = normalize(query_vec)
        scores, indices = self.index.search(query_vec, top_k)
        return [(self.codes[i], scores[0][rank]) for rank, i in enumerate(indices[0])]
