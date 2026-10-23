"""Retrieve the most relevant NLP topics using TF-IDF and cosine similarity."""

import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the knowledge base once when the application starts.
with open("database.json", encoding="utf-8") as file:
    database = json.load(file)

texts = [
    item["topic"] + " " + item["summary"] + " " + " ".join(item["keywords"])
    for item in database
]

vectorizer = TfidfVectorizer(stop_words="english")
document_vectors = vectorizer.fit_transform(texts)


def retrieve(question, top_k=3):
    """Return the top matching topics (or [] when there is no word overlap)."""
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, document_vectors)[0]
    best_indices = scores.argsort()[::-1][:top_k]
    return [database[i] for i in best_indices if scores[i] > 0]
