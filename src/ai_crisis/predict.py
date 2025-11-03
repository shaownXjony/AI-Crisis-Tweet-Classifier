# src/ai_crisis/predict.py
import numpy as np

def sigmoid(x):
    """Compute sigmoid for confidence scaling."""
    return 1 / (1 + np.exp(-x))

def predict_text(model, vectorizer, text, clean_func):
    """
    Predict the class and confidence for a single tweet.
    - Uses model.predict_proba if available, otherwise uses sigmoid(decision_function)
    """
    if not text or not isinstance(text, str):
        return "invalid_input", 0.0

    # Clean and transform text
    cleaned = clean_func(text)
    X = vectorizer.transform([cleaned])

    # Prediction
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        idx = probs.argmax(axis=1)[0]
        pred = model.classes_[idx] if hasattr(model, "classes_") else idx
        conf = float(probs[0, idx])
    elif hasattr(model, "decision_function"):
        score = model.decision_function(X)[0]
        conf = float(sigmoid(score))
        pred = model.predict(X)[0]
    else:
        pred = model.predict(X)[0]
        conf = 0.5  # unknown confidence

    # Clip for display (avoid 0.000)
    conf = max(min(conf, 1 - 1e-6), 1e-6)

    return pred, conf
