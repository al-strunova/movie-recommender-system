from pathlib import Path
import pickle
from src.config import config



class PopularityModel:
    """Calculates a popularity score for all movies."""

    def __init__(self):
        self.movies_with_scores = None
        self.m = None  # minimum votes threshold
        self.C = None  # mean rating across all movies
        self.m_quantile_used = None

    def fit(self, m_quantile: float = None):
        """Calculate weighted ratings and create popularity ranking."""
        print("Fitting PopularityModel...")
        from src.features.feature_engineering import FeatureEngineering

        builder = FeatureEngineering()
        self.movies_with_scores, self.m, self.C = builder.build_scored_movies()
        self.m_quantile_used = m_quantile if m_quantile is not None else config.params.recommender.popularity_m_quantile

        print("PopularityModel is ready.")
        print(f"Fitted with parameters: C={self.C:.2f}, m={self.m:.0f}, quantile={self.m_quantile_used}")

    def get_top_recommendations(self, user_id: int,
                                n: int = config.params.recommender.n_final_recommendations,
                                exclude_seen: list = None):
        """Return top n movies. Same recommendations for all users"""
        if self.movies_with_scores is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Start with the pre-sorted list of popular movies
        recs_df = self.movies_with_scores
        if recs_df.empty:
            return []

        # If there's a list of seen movies, filter them out efficiently
        if exclude_seen is not None:
            recs_df = recs_df[~recs_df['movieId'].isin(exclude_seen)]

        # Get the top n results from the filtered DataFrame
        top_n = recs_df.head(n)

        # Return the result in the desired format
        return list(zip(top_n['movieId'], top_n['weighted_rating']))

    def get_top_weighted_recommendations(self, user_id: int,
                                         n: int = config.params.recommender.n_final_recommendations,
                                         exclude_seen: list = None):
        """
        Return a random sample of n movies, weighted by their rating.
        This provides more discovery than the static top-n list.
        """
        if self.movies_with_scores is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Start with the pre-sorted list of popular movies
        recs_df = self.movies_with_scores
        if recs_df.empty:
            return []

        # If there's a list of seen movies, filter them out efficiently
        if exclude_seen is not None:
            recs_df = recs_df[~recs_df['movieId'].isin(exclude_seen)]

        # Use .sample() with the 'weighted_rating' column as weights
        random_recs = recs_df.sample(n=min(n, len(recs_df)), weights='weighted_rating')

        return list(zip(random_recs['movieId'], random_recs['weighted_rating']))

    def save(self, path: Path = None):
        path = path or config.paths.POPULARITY_MODEL_PATH
        model_data = {
            'movies_with_scores': self.movies_with_scores,
            'm': self.m,
            'C': self.C
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        print("Popularity model saved successfully.")

    @classmethod
    def load(cls, path: Path = None):
        path = path or config.paths.POPULARITY_MODEL_PATH
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        model = cls()
        model.movies_with_scores = model_data['movies_with_scores']
        model.m = model_data['m']
        model.C = model_data['C']

        print("Popularity model loaded successfully.")
        return model
