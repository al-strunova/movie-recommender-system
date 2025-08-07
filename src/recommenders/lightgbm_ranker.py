import pickle
import lightgbm as lgb
from src.config import config


class LightGBMRanker:

    def __init__(self):
        self.model = None

    def fit(self, X=None, y=None):
        """
        Train the model. If X, y not provided, builds features automatically.

        Args:
            X: Feature DataFrame (optional - will build if not provided)
            y: Target labels (optional - will build if not provided)
        """
        if X is None or y is None:
            from src.features.feature_engineering import FeatureEngineering
            print("Building features for LightGBM training...")
            fe = FeatureEngineering()
            X, y, metadata = fe.build_ranking_dataset()
            print(f"Built features: {X.shape}")
        self.model = lgb.LGBMClassifier(**config.params.lgb_ranking.lgb_params)
        self.model.fit(X, y)
        print("LightGBM training complete")

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)

    def save(self, path=None):
        """Save the model."""
        if path is None:
            path = config.paths.LIGHTGBM_RANKER_PATH
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print("LightGMB Model saved successfully")

    @classmethod
    def load(cls, path=None):
        """Load the model."""
        if path is None:
            path = config.paths.LIGHTGBM_RANKER_PATH
        instance = cls()
        try:
            with open(path, 'rb') as f:
                instance.model = pickle.load(f)
            print(f"LightGBM loaded successfully")
            return instance
        except Exception as e:
            print(f"Failed to load LightGBM from {path}: {e}")
            raise

