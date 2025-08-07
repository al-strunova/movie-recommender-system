import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from src.config import config
from src.data.data_loader import DataLoader
from src.recommenders.faiss_recommender_abc import FaissEmbeddingRecommenderABC


class ContentBasedRecommender(FaissEmbeddingRecommenderABC):
    """
    A content-based recommender that uses TFID and SVC for movie content embeddings and Faiss
    for high-performance similarity search.
    """

    def __init__(self, n_components: int = config.params.content.svd_components):
        super().__init__(n_components)

    def _get_init_params(self):
        return {'n_components': self.n_components}

    def fit(self):
        """Trains the model by performing the full TF-IDF -> SVD pipeline + Faiss search index"""
        print(f"Fitting {self.__class__.__name__}...")
        import faiss

        # Build Movie Embeddings
        from src.features.feature_engineering import FeatureEngineering
        feature_builder = FeatureEngineering()
        movie_documents = feature_builder.build_movie_documents()

        print("Fitting TF-IDF and SVD...")
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=config.params.content.tfidf_max_features)
        tfidf_matrix = tfidf_vectorizer.fit_transform(movie_documents)

        svd_model = TruncatedSVD(n_components=self.n_components, random_state=42)
        movie_embeddings_raw = svd_model.fit_transform(tfidf_matrix)

        # Build the High-Performance Faiss Index
        print("Normalizing embeddings and building Faiss index...")
        self.movie_embeddings = movie_embeddings_raw / np.linalg.norm(movie_embeddings_raw, axis=1, keepdims=True)
        self.movie_embeddings = self.movie_embeddings.astype('float32')

        # Create the Faiss index. IndexFlatIP is for Inner Product (which equals cosine similarity for normalized vectors)
        self.faiss_index = faiss.IndexFlatIP(self.n_components)
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

        print("Fit complete.")
