from src.recommenders.popularity_model import PopularityModel
from src.recommenders.genre_popularity_recommender import GenrePopularityRecommender
from src.recommenders.collaborative_model import CollaborativeFilteringModel
from src.recommenders.content_based_recommender import ContentBasedRecommender
from src.recommenders.semantic_recommender import SemanticRecommender


def train_all_models():
    """Train and save all candidate models."""

    models = [
        PopularityModel(),
        GenrePopularityRecommender(),
        CollaborativeFilteringModel(),
        ContentBasedRecommender(),
        SemanticRecommender()
    ]

    for model in models:
        print(f"\nTraining {model.__class__.__name__}...")
        model.fit()
        model.save()
        print(f"✓ {model.__class__.__name__} saved")

    print("\n✓ All models trained and saved!")


if __name__ == "__main__":
    train_all_models()
