import gc
import pickle
import implicit
from scipy.sparse import csr_matrix, save_npz, load_npz

from src.data.data_loader import DataLoader
from src.config import config


class CollaborativeFilteringModel:

    def __init__(self,
                 factors=config.params.als.factors,
                 regularization=config.params.als.regularization,
                 iterations=config.params.als.iterations):
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations
        )
        self.user_item_matrix = None
        self.user_map = None
        self.movie_map = None
        self.inverse_movie_map = None

    def fit(self):
        loader = DataLoader()
        ratings_df = loader.get_ratings()

        # Create mappings from original IDs to matrix indices
        self.user_map = {id: i for i, id in enumerate(ratings_df['userId'].unique())}
        self.movie_map = {id: i for i, id in enumerate(ratings_df['movieId'].unique())}
        self.inverse_movie_map = {i: id for id, i in self.movie_map.items()}

        # Map original IDs to new matrix indices
        ratings_df['user_idx'] = ratings_df['userId'].map(self.user_map)
        ratings_df['movie_idx'] = ratings_df['movieId'].map(self.movie_map)

        # Create the sparse user-item matrix
        self.user_item_matrix = csr_matrix((ratings_df['rating'], (ratings_df['user_idx'], ratings_df['movie_idx'])))

        print("Fitting Collab ALS model...")
        self.model.fit(self.user_item_matrix)
        print("Collab ALS model fit complete.")

    def recommend(self, user_id: int,
                  n: int = config.params.recommender.n_final_recommendations,
                  exclude_seen: set = None) -> list:
        if user_id not in self.user_map:
            return []  # Cannot recommend for a user not in the training data

        # Map the original user ID to our internal matrix index
        user_idx = self.user_map[user_id]

        # Use the model's recommend method. It needs the user's row from the training matrix.
        # N=n+len(exclude_seen) to get extra items in case some are filtered out.
        n_to_fetch = n + (len(exclude_seen) if exclude_seen is not None else 0)
        indices, scores = self.model.recommend(user_idx,
                                               self.user_item_matrix[user_idx],
                                               N=n_to_fetch,
                                               filter_already_liked_items=False)
        recommendations = list(zip(indices, scores))

        # Convert the recommendations back to original movie IDs and format the output
        final_recs = []
        for movie_idx, score in recommendations:
            # Stop as soon as we have collected n recommendations.
            if len(final_recs) >= n:
                break

            original_movie_id = self.inverse_movie_map[movie_idx]
            if exclude_seen and original_movie_id in exclude_seen:
                continue
            final_recs.append((original_movie_id, score))

        return final_recs

    def predict_score(self, user_id: int, movie_id: int) -> float:
        """Predicts a single score for a given user/movie pair."""
        # Map from original IDs to internal indices
        if user_id not in self.user_map or movie_id not in self.movie_map:
            return 0.0  # Return a neutral score if we can't predict

        user_idx = self.user_map[user_id]
        movie_idx = self.movie_map[movie_id]

        # Get the embedding vectors from the trained model
        user_vector = self.model.user_factors[user_idx]
        movie_vector = self.model.item_factors[movie_idx]

        # Calculate the dot product
        return user_vector.dot(movie_vector)

    def save(self):

        # Use the paths from the Config file
        model_path = config.paths.ALS_MODEL_PATH
        maps_path = config.paths.ALS_MAPS_PATH

        self.model.save(model_path)
        save_npz(config.paths.ALS_MATRIX_PATH, self.user_item_matrix)

        # We save the python dictionaries with pickle
        with open(maps_path, 'wb') as f:
            pickle.dump({
                'user_map': self.user_map,
                'movie_map': self.movie_map,
                'inverse_movie_map': self.inverse_movie_map
            }, f)
        print("Collab ALS model, matrix, and maps saved successfully.")

    def cleanup(self):
        """Release memory and resources"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'user_item_matrix'):
            del self.user_item_matrix
        gc.collect()

    @classmethod
    def load(cls):
        """
        Loads a pre-trained model and mappings from paths defined in config.
        No longer needs a path argument.
        """

        # Load mappings and the user-item matrix first
        with open(config.paths.ALS_MAPS_PATH, 'rb') as f:
            maps = pickle.load(f)
        user_item_matrix = load_npz(config.paths.ALS_MATRIX_PATH)

        loaded_als_model = implicit.als.AlternatingLeastSquares().load(config.paths.ALS_MODEL_PATH)

        # Create an instance of our own class
        model = cls()

        # Populate our instance with all the loaded components
        model.model = loaded_als_model
        model.user_item_matrix = user_item_matrix
        model.user_map = maps['user_map']
        model.movie_map = maps['movie_map']
        model.inverse_movie_map = maps['inverse_movie_map']

        print("Collab ALS model loaded successfully.")
        return model
