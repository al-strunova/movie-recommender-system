import gc
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import config
from src.data.data_loader import DataLoader
from src.recommenders.faiss_recommender_abc import FaissEmbeddingRecommenderABC



class SemanticRecommender(FaissEmbeddingRecommenderABC):

    def __init__(self, model_name: str = config.params.content.sentence_transformer_model):
        super().__init__()
        self.model_name = model_name
        self.sentence_model = SentenceTransformer(self.model_name)
        self.user_taste_vectors = None

    def _get_init_params(self):
        return {'model_name': self.model_name}

    def fit(self):
        """
        "Trains" the model by generating semantic embeddings for all movies
        and building a Faiss index for fast search.
        """
        print(f"Fitting {self.__class__.__name__} with model '{self.model_name}'...")

        import faiss

        # Get the movie "documents" from your feature engineering class
        from src.features.feature_engineering import FeatureEngineering
        feature_builder = FeatureEngineering()
        movie_documents = feature_builder.build_movie_documents(semantic_format=True)

        # Create the embeddings
        print(f"Creating embeddings for {len(movie_documents)} documents...")
        self.movie_embeddings = self.sentence_model.encode(
            movie_documents.tolist(),
            batch_size=config.params.content.sentence_transformer_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Build the Faiss index from these new embeddings
        embedding_dimension = self.movie_embeddings.shape[1]
        print(f"Embeddings created with dimension: {embedding_dimension}")
        self.faiss_index = faiss.IndexFlatIP(embedding_dimension)
        self.faiss_index.add(self.movie_embeddings.astype('float32'))
        print(f"Faiss index built with {self.faiss_index.ntotal} vectors.")

        # Create ID Mappings and Pre-process User Likes
        print("Creating ID maps and pre-processing user likes...")
        self.movie_id_map = {movie_id: i for i, movie_id in enumerate(movie_documents.index)}
        self.inverse_movie_id_map = {i: movie_id for movie_id, i in self.movie_id_map.items()}

        # Build a df of users and the movies they previously liked
        loader = DataLoader()
        ratings_df = loader.get_ratings()
        liked_ratings_df = ratings_df[ratings_df['rating'] >= 4.0]
        self.user_likes = liked_ratings_df.groupby('userId')['movieId'].apply(list).to_dict()

        self.movies_with_weighted_score = feature_builder.build_scored_movies()[0]

        self.user_taste_vectors = {}
        for user_id, liked_m_ids in self.user_likes.items():
            liked_indices = [self.movie_id_map[mid] for mid in liked_m_ids if mid in self.movie_id_map]
            if liked_indices:
                liked_vectors = self.movie_embeddings[liked_indices]
                self.user_taste_vectors[user_id] = np.mean(liked_vectors, axis=0)

        print("Fit complete.")

    def predict_score(self, user_id: int, movie_id: int) -> float:
        """Predicts a single content similarity score for a user/movie pair."""
        if user_id not in self.user_taste_vectors or movie_id not in self.movie_id_map:
            return 0.0

        # Get the pre-computed vectors
        user_vector = self.user_taste_vectors[user_id].reshape(1, -1)
        movie_idx = self.movie_id_map[movie_id]
        movie_vector = self.movie_embeddings[movie_idx].reshape(1, -1)

        # Calculate and return the cosine similarity
        return cosine_similarity(user_vector, movie_vector)[0][0]

    def save(self, filepath=None):

        import faiss

        if filepath is None:
            filepath = config.paths.CANDIDATE_MODEL_DIR / f"{self.__class__.__name__.lower()}.pkl"

        # Save FAISS index
        faiss_index_path = filepath.with_suffix('.faiss')
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(faiss_index_path))

        state = {
            'init_params': self._get_init_params(),
            'movie_embeddings': self.movie_embeddings,
            'user_likes': self.user_likes,
            'movie_id_map': self.movie_id_map,
            'inverse_movie_id_map': self.inverse_movie_id_map,
            'movies_with_weighted_score': self.movies_with_weighted_score,
            'user_taste_vectors': self.user_taste_vectors
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        print(f"Semantic model saved successfully")

    @classmethod
    def load(cls, filepath=None):
        """Override to recreate sentence_model."""

        import faiss

        if filepath is None:
            filepath = config.paths.CANDIDATE_MODEL_DIR / f"{cls.__name__.lower()}.pkl"

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # Create instance (this will create sentence_model)
        instance = cls(**state['init_params'])

        # Restore all attributes except init_params
        for key, value in state.items():
            if key != 'init_params':
                setattr(instance, key, value)

        # Load FAISS index
        faiss_index_path = filepath.with_suffix('.faiss')
        if faiss_index_path.exists():
            instance.faiss_index = faiss.read_index(str(faiss_index_path))

        print(f"Semantic model loaded successfully")
        return instance

    def cleanup(self):
        """Release memory and resources"""
        if hasattr(self, 'faiss_index'):
            del self.faiss_index
        if hasattr(self, 'sentence_model'):
            del self.sentence_model
        if hasattr(self, 'movie_embeddings'):
            del self.movie_embeddings
        # Clear any other large objects
        gc.collect()
