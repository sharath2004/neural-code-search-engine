from flask import Flask, request, jsonify
from app.search import CodeSearchEngine

app = Flask(__name__)
engine = CodeSearchEngine()

@app.route("/search")
def search():
    q = request.args.get("q", "")
    if not q:
        return jsonify({"error": "Query cannot be empty"}), 400

    results = engine.search(q)
    return jsonify([
        {"code": code, "score": float(score)}
        for code, score in results
    ])

if __name__ == "__main__":
    app.run(debug=True)
