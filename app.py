import os
import joblib
import numpy as np
from flask import Flask, request, jsonify
from scipy.sparse import hstack
from textblob import TextBlob

app = Flask(__name__)

# ---------- load fitted artefacts ----------
vectorizer = joblib.load("tfidf_vectorizer.joblib")
svd        = joblib.load("svd_transformer.joblib")
ensemble   = joblib.load("fake_news_ensemble_model.joblib")

# ---------- preprocessing ----------
def preprocess_and_stem(text: str) -> str:
    """Clean & stem text (same logic as Colab)."""
    import re
    text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------- routes ----------
@app.route("/")
def home():
    return {"message": "Fake-News detector API – POST to /predict"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        news = request.json["text"]
        if not news:
            return jsonify({"error": "empty text"}), 400

        proc  = preprocess_and_stem(news)
        xv    = vectorizer.transform([proc])
        extra = np.array([[TextBlob(proc).sentiment.polarity, len(proc)]])
        x_full = hstack([xv, extra])
        x_final = svd.transform(x_full)

        pred  = ensemble.predict(x_final)[0]
        proba = ensemble.predict_proba(x_final)[0]

        return jsonify({
            "label":  "FAKE" if pred == 0 else "REAL",
            "confidence": {"fake": round(float(proba[0]), 3),
                          "real": round(float(proba[1]), 3)}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)