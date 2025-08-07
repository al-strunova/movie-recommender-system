import numpy as np
import pickle
from sentence_transformers import CrossEncoder
from src.config import config


class CrossEncoderRanker:

    def __init__(self):
        self.model = CrossEncoder(config.params.cross_encoder.model_name)
        print(f"CrossEncoder model loaded successfully")

    def predict_proba(self, user_texts, movie_texts):
        if self.model is None:
            raise ValueError("Model not trained yet")

        text_pairs = [[u, m] for u, m in zip(user_texts, movie_texts)]
        scores = self.model.predict(text_pairs)

        # Convert to probabilities and return in [neg, pos] format
        proba_pos = 1 / (1 + np.exp(-np.clip(scores, -500, 500)))
        proba_neg = 1 - proba_pos
        return np.column_stack([proba_neg, proba_pos])