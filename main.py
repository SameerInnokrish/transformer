from flask import Flask, jsonify, make_response, Response, render_template, request
import json
from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np


app = Flask(__name__)

with open('data/data.json', 'r', encoding='utf-8') as file:
    documents=json.load(file)

embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

corpus = [d['abstract'] for d in documents]

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)


r = corpus_embeddings.cpu().numpy()

index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
index.add_with_ids(np.array(r), np.array(range(0, len(corpus))).astype(np.int64))

@app.route('/')
def home():
    return render_template('search.html')

@app.route('/search/results', methods=['GET', 'POST'])
def search_request():
    search_term = request.form["input"]
    k = 20
    encoded_query = embedder.encode([search_term])
    top_k = index.search(encoded_query, k)
    answers = [corpus[_id] for _id in top_k[1].tolist()[0]]

    return render_template('results.html', answers=answers)

if __name__ == '__main__':
    app.secret_key = 'mysecret'
    app.run(host='0.0.0.0', debug=True)


if app.config["DEBUG"]:
    @ app.after_request
    def after_request(response):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
        response.headers["Expires"] = 0
        response.headers["Pragma"] = "no-cache"
        return response
