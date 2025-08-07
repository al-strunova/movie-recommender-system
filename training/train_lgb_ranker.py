from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from src.config import config
from src.features.feature_engineering import FeatureEngineering
from src.recommenders.lightgbm_ranker import LightGBMRanker


def train_lgb_ranker():
    """Train LightGBM ranker"""

    print("Training LightGBM Ranker...")

    # 1. Get data
    fe = FeatureEngineering()
    X, y, metadata = fe.build_ranking_dataset()

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.params.lgb_ranking.test_size,
        random_state=config.params.lgb_ranking.random_state
    )

    # 3. Train
    ranker = LightGBMRanker()
    ranker.fit(X_train, y_train)

    # 4. Evaluate
    y_pred_proba = ranker.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    print(f"\nTest Metrics:")
    print(f"AUC:       {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")

    # 5. Save
    ranker.save()


if __name__ == "__main__":
    train_lgb_ranker()
