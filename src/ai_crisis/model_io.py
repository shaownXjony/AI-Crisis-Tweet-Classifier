# src/ai_crisis/model_io.py
import os
import joblib
import logging

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def load_model_and_vectorizer(model_path: str, vect_path: str):
    """
    Load model and vectorizer safely.
    Returns (model, vectorizer)
    """
    model, vect = None, None

    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            LOGGER.info(f"Loaded model from {model_path}")
        except Exception as e:
            LOGGER.error(f"Error loading model: {e}")
    else:
        LOGGER.warning(f"Model file not found at {model_path}")

    if os.path.exists(vect_path):
        try:
            vect = joblib.load(vect_path)
            LOGGER.info(f"Loaded vectorizer from {vect_path}")
        except Exception as e:
            LOGGER.error(f"Error loading vectorizer: {e}")
    else:
        LOGGER.warning(f"Vectorizer file not found at {vect_path}")

    return model, vect
