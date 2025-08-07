import pickle
import pandas as pd
from src.config import config
from abc import ABC, abstractmethod


class FaissEmbeddingRecommenderABC(ABC):
    """
    An abstract base class for all recommenders that work by finding
    similarities between item embeddings.

    It contains the shared logic for making recommendations once embeddings
    and a Faiss index have been created.
    """

    def __init__(self, n_components: int | None = None):
        self.n_components = n_components
        self.movie_embeddings = None
        self.faiss_index = None
        self.user_likes = None
        self.movie_id_map = None
        self.inverse_movie_id_map = None
        self.movies_with_weighted_score = None

    @abstractmethod
    def _get_init_params(self):
        """Each subclass must define what params it needs for initialization."""
        raise NotImplementedError

    @abstractmethod
    def fit(self):
        """
        This is an abstract method. Each child class MUST implement its own
        version of fit(), as this is what makes them unique.
        """
        raise NotImplementedError("Each recommender must implement its own fit method.")

    def _get_similar_movies(self, movie_id: int,
                            n: int = config.params.recommender.n_candidates_per_movie) -> list:
        """Finds similar movies using the high-performance Faiss index."""
        if movie_id not in self.movie_id_map:
            return []
        movie_idx = self.movie_id_map[movie_id]
        query_vector = self.movie_embeddings[movie_idx].reshape(1, -1)
        distances, indices = self.faiss_index.search(query_vector, n + 1)

        similar_movies = []
        for i, dist in zip(indices[0], distances[0]):
            if i == -1 or i == movie_idx: continue  # -1 indicates an empty slot
            similar_movies.append((self.inverse_movie_id_map[i], dist))

        return similar_movies

    def _generate_candidates(self, user_id: int) -> dict:
        """Helper to generate a pool of candidates based on user's top likes."""
        user_top_movies = self.user_likes.get(user_id, [])[:config.params.recommender.n_seed_movies]
        if not user_top_movies:
            return {}

        all_candidates = {}
        for movie_id in user_top_movies:
            similar_movies = self._get_similar_movies(movie_id)
            for sim_movie_id, sim_score in similar_movies:
                all_candidates[sim_movie_id] = all_candidates.get(sim_movie_id, 0) + sim_score
        return all_candidates

    def _filter_seen_and_get_top_n(self, sorted_candidates: list, user_id: int, n: int, exclude_seen: set) -> list:
        """Helper to perform the final filtering and slicing."""
        if exclude_seen is None:
            exclude_seen = set(self.user_likes.get(user_id, []))

        final_recommendations = []
        for movie_id, score in sorted_candidates:
            if len(final_recommendations) >= n:
                break
            if movie_id not in exclude_seen:
                final_recommendations.append((movie_id, float(score)))
        return final_recommendations

    def recommend_by_similarity(self,
                                user_id: int,
                                n: int = config.params.recommender.n_final_recommendations,
                                exclude_seen: set = None) -> list:
        """Ranks candidates purely by content similarity score."""
        if self.user_likes is None: raise RuntimeError("Model not fitted.")

        all_candidates = self._generate_candidates(user_id)
        if not all_candidates: return []

        sorted_candidates = sorted(all_candidates.items(), key=lambda item: item[1], reverse=True)
        return self._filter_seen_and_get_top_n(sorted_candidates, user_id, n, exclude_seen)

    def recommend_by_hybrid_score(self, user_id: int,
                                  n: int = config.params.recommender.n_final_recommendations,
                                  exclude_seen: set = None) -> list:
        """
        Generates recommendations using a hybrid score of content similarity
        and item popularity for a more robust ranking.
        """
        if self.user_likes is None or self.movies_with_weighted_score is None:
            raise RuntimeError("Model is not fitted. Please call fit() before recommending.")

        all_candidates = self._generate_candidates(user_id)
        if not all_candidates: return []

        candidates_df = pd.DataFrame(all_candidates.items(), columns=['movieId', 'similarity_score'])
        candidates_df = candidates_df.merge(self.movies_with_weighted_score[['movieId', 'weighted_rating']],
                                            on='movieId',
                                            how='left').fillna(0)

        sim_max = candidates_df['similarity_score'].max()
        pop_max = candidates_df['weighted_rating'].max()
        if sim_max > 0: candidates_df['similarity_score'] /= sim_max
        if pop_max > 0: candidates_df['weighted_rating'] /= pop_max

        alpha = config.params.recommender.similarity_score_weight
        beta = config.params.recommender.popularity_score_weight
        candidates_df['final_score'] = (alpha * candidates_df['similarity_score']) + (
                beta * candidates_df['weighted_rating'])

        sorted_candidates_df = candidates_df.sort_values('final_score', ascending=False)
        sorted_candidates_list = list(zip(sorted_candidates_df['movieId'], sorted_candidates_df['final_score']))
        return self._filter_seen_and_get_top_n(sorted_candidates_list, user_id, n, exclude_seen)

    def save(self, filepath=None):
        """Save the model to disk."""
        import faiss

        if filepath is None:
            filepath = config.paths.CANDIDATE_MODEL_DIR / f"{self.__class__.__name__.lower()}.pkl"

        init_params = self._get_init_params()

        faiss_index_path = filepath.with_suffix('.faiss')
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(faiss_index_path))

        # Save everything else
        state = {
            'init_params': init_params,
            'movie_embeddings': self.movie_embeddings,
            'user_likes': self.user_likes,
            'movie_id_map': self.movie_id_map,
            'inverse_movie_id_map': self.inverse_movie_id_map,
            'movies_with_weighted_score': self.movies_with_weighted_score
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        print(f"{self.__class__.__name__.lower()} model saved successfully.")

    @classmethod
    def load(cls, filepath=None):
        """Load the model from disk."""
        import faiss

        if filepath is None:
            filepath = config.paths.CANDIDATE_MODEL_DIR / f"{cls.__name__.lower()}.pkl"

        # Load state
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # Create instance with correct init params
        instance = cls(**state['init_params'])

        # Restore state
        for key, value in state.items():
            if key != 'init_params':
                setattr(instance, key, value)

        faiss_index_path = filepath.with_suffix('.faiss')
        if faiss_index_path.exists():
            instance.faiss_index = faiss.read_index(str(faiss_index_path))

        print(f"{cls.__name__.lower()} model loaded successfully.")

        return instance

    def cleanup(self):
        """Release memory."""
        attrs_to_delete = [
            'movie_embeddings', 'faiss_index', 'user_likes',
            'movie_id_map', 'inverse_movie_id_map',
            'movies_with_weighted_score'
        ]

        for attr in attrs_to_delete:
            if hasattr(self, attr):
                delattr(self, attr)
