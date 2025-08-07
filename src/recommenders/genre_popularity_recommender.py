import pickle
from pathlib import Path
from src.config import config


class GenrePopularityRecommender:
    """A personalized recommender that uses pre-computed user profiles (top genre)
    and pre-computed lists of popular movies per genre."""

    def __init__(self):
        self.user_profiles = None
        self.top_movies_per_genre = None

    def fit(self):
        """
        Fits the model by orchestrating the creation of all necessary features
        from the FeatureEngineering class.
        """
        print(f"Fitting {self.__class__.__name__}...")

        # 1. Instantiate the feature factory to get our data products
        from src.features.feature_engineering import FeatureEngineering
        feature_builder = FeatureEngineering()

        # 2. Call the builder to get the features this model needs.
        #    This is where the heavy work happens.
        self.user_profiles = feature_builder.build_user_genre_profiles()
        self.top_movies_per_genre = feature_builder.build_top_movies_per_genre_cache()

        print("Fit complete. Genre popularity model is ready to recommend.")
        return self  # Return self to allow for chaining, e.g., model.fit().recommend()

    def recommend(self, user_id: int,
                  n: int = config.params.recommender.n_final_recommendations,
                  exclude_seen: set = None) -> list:
        """
        Return top n movies for a user with a user favorite genre
        """
        if self.user_profiles is None or self.top_movies_per_genre is None:
            raise RuntimeError("The model has not been fitted yet. Please call the .fit() method first.")

        top_genre = self.user_profiles.get(user_id)
        if not top_genre:
            return []

        ranked_movies = self.top_movies_per_genre.get(top_genre, [])
        if not ranked_movies:
            return []

        if not exclude_seen:
            return ranked_movies[:n]

        final_recommendations = []
        for mid, score in ranked_movies:
            if len(final_recommendations) >= n:
                break
            if mid not in exclude_seen:
                final_recommendations.append((mid, score))

        return final_recommendations

    def save(self, path: Path = None):

        # Use a default path from Config if none is provided
        path = path or config.paths.GENRE_MODEL_PATH
        model_state = {
            'user_profiles': self.user_profiles,
            'top_movies_per_genre': self.top_movies_per_genre
        }
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)

        print("Genre popularity model saved successfully.")

    @classmethod
    def load(cls, path: Path = None):

        path = path or config.paths.GENRE_MODEL_PATH

        with open(path, 'rb') as f:
            model_state = pickle.load(f)

        # 1. Create a new instance of the class, fulfilling the __init__ dependency
        model = cls()

        # 2. "Hydrate" the instance with the saved state
        model.user_profiles = model_state['user_profiles']
        model.top_movies_per_genre = model_state['top_movies_per_genre']

        print("Genre popularity model loaded successfully.")
        return model
