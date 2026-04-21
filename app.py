# app.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import dill as pickle

#---------------- Preprocess and NLP setup ----------------
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('stopwords')

#---------------- Define combined model class ----------------
class SpamModelWithRecommendation:
    def __init__(self, model, vectorizer, unigrams, bigrams):
        self.model = model
        self.vectorizer = vectorizer
        self.unigrams = unigrams
        self.bigrams = bigrams
        self.V = len(unigrams)
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

    # Preprocess method inside the class
    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words]
        tokens = [self.stemmer.stem(t) for t in tokens]
        return " ".join(tokens)

    # Spam/Ham Prediction
    def predict(self, texts):
        X_vec = self.vectorizer.transform(texts)
        return self.model.predict(X_vec)

    def predict_proba(self, texts):
        X_vec = self.vectorizer.transform(texts)
        return self.model.predict_proba(X_vec)

    # Recommendation System
    def bigram_probability(self, w1, w2):
        bigram_count = self.bigrams.get((w1, w2), 0)
        unigram_count = self.unigrams.get((w1,), 0)
        return (bigram_count + 1) / (unigram_count + self.V)

    def recommend_next_word(self, current_word, top_k=5):
        candidates = []
        for (w1, w2), count in self.bigrams.items():
            if w1 == current_word:
                prob = self.bigram_probability(w1, w2)
                candidates.append((w2, prob))
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    def complete_sentence(self, sentence, top_k=5):
        sentence = self.preprocess(sentence)
        words = sentence.split()
        if not words:
            return []
        last_word = words[-1]
        return [w for w, _ in self.recommend_next_word(last_word, top_k)]

#---------------- Load the combined model ----------------
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

with open("combined_spam_recommendation_model.pkl", "rb") as f:
    model = pickle.load(f)

#---------------- Request schemas ----------------
class DetectRequest(BaseModel):
    text: str

class RecommendRequest(BaseModel):
    text: str
    top_k: int = 5

#---------------- FastAPI app ----------------
app = FastAPI(title="Spam Detection + Recommendation API")

#---------------- Spam/Ham Detection Endpoint ----------------
@app.post("/detect")
def detect_spam(message: DetectRequest):
    label = model.predict([message.text])[0]
    return JSONResponse({"prediction": label})

#---------------- Recommendation Endpoint ----------------
@app.post("/recommend")
def recommend_words(message: RecommendRequest):
    suggestions = model.complete_sentence(message.text, top_k=message.top_k)
    return JSONResponse({"suggestions": suggestions})

#---------------- Root ----------------
@app.get("/")
def root():
    return {"message": "Spam Detection API and Recommendation API running."}
