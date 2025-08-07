import pickle
import random
from typing import List, Tuple, Optional, Callable
from src.config import config
from src.features.feature_engineering import FeatureEngineering


class TwoStageRecommender:
    """Two-stage recommendation system that combines candidate generation with reranking."""

    def __init__(self, candidate_models=None, collab_model=None, semantic_model=None,
                 use_cross_encoder=False):
        """
        Initialize two-stage recommender.

        Args:
            candidate_models: List of (model, method_name) tuples for candidate generation
            collab_model: Collaborative filtering model for feature engineering
            semantic_model: Semantic model for feature engineering
            use_cross_encoder: If True, use CrossEncoder for reranking; if False, use LightGBM
        """
        self.use_cross_encoder = use_cross_encoder
        self.candidate_models = candidate_models or []

        self.ranker = None

        # Load ranker
        if self.use_cross_encoder:
            try:
                from src.recommenders.cross_encoder_ranker import CrossEncoderRanker
                self.ranker = CrossEncoderRanker()
            except Exception as e:
                print(f"Failed to load CrossEncoder: {e}")
                raise ValueError(f"Failed to load CrossEncoder: {e}")

        if not self.use_cross_encoder:
            try:
                from src.recommenders.lightgbm_ranker import LightGBMRanker
                self.ranker = LightGBMRanker.load()
            except Exception as e:
                print(f"Failed to load new LightGBM: {e}")
                raise ValueError(f"Failed to load LightGBM: {e}")

        self.fe = FeatureEngineering()
        extracted_collab = None
        extracted_semantic = None

        for model, method_name in self.candidate_models:
            if model.__class__.__name__ == 'CollaborativeFilteringModel':
                extracted_collab = model
            elif model.__class__.__name__ == 'SemanticRecommender':
                extracted_semantic = model

        self.collab_model = extracted_collab or collab_model
        self.semantic_model = extracted_semantic or semantic_model

        # Load fresh only if neither extraction nor passing worked
        if self.collab_model is None:
            try:
                from src.recommenders.collaborative_model import CollaborativeFilteringModel
                self.collab_model = CollaborativeFilteringModel.load()
            except:
                self.collab_model = None

        if self.semantic_model is None:
            try:
                from src.recommenders.semantic_recommender import SemanticRecommender
                self.semantic_model = SemanticRecommender.load()
            except:
                self.semantic_model = None

    def get_candidates_description(self):
        """Get a description of the candidate models used."""
        if not self.candidate_models:
            return "No candidate models"

        model_names = []
        for model, method in self.candidate_models:
            model_names.append(f"{model.__class__.__name__}.{method}")

        ranker_type = "CrossEncoder" if self.use_cross_encoder else "LightGBM"
        return f"Candidates: {', '.join(model_names)}. Ranker: {ranker_type}"

    def get_candidates(self, user_id: int, n_candidates: int = None) -> List[Tuple[int, float]]:
        """Get candidate movies from the provided models."""

        if n_candidates is None:
            n_candidates = config.params.two_stage.n_candidates

        if not self.candidate_models:
            print("No candidate models provided")
            return []

        all_candidates = set()
        candidates_per_model = max(1, n_candidates // len(self.candidate_models))

        for model, method_name in self.candidate_models:
            try:
                recommend_method = getattr(model, method_name)
                candidates = recommend_method(user_id, n=candidates_per_model)
                all_candidates.update(candidates)
            except Exception as e:
                print(f"Model {model.__class__.__name__} failed: {e}")

        # Remove duplicates and shuffle
        unique_candidates = {}
        for movie_id, score in all_candidates:
            if movie_id not in unique_candidates:
                unique_candidates[movie_id] = score

        candidate_list = list(unique_candidates.items())
        random.shuffle(candidate_list)

        return candidate_list[:n_candidates]

    def recommend(self, user_id: int, n: int = 10, exclude_seen: set = None) -> List[tuple]:
        """Get recommendations for a user."""

        # Get candidates
        candidates = self.get_candidates(user_id)
        if not candidates:
            print("No candidates - returning empty")
            return []

        if exclude_seen:
            candidates = [(movie_id, score) for movie_id, score in candidates if movie_id not in exclude_seen]

        if not candidates:
            print("No candidates after filtering - returning empty")
            return []

        # Rerank
        candidate_movie_ids = [movie_id for movie_id, score in candidates]

        if self.use_cross_encoder:
            # CrossEncoder reranking
            try:
                user_texts, movie_texts = self.fe.build_crossencoder_inference_data([user_id], candidate_movie_ids)
                scores = self.ranker.predict_proba(user_texts, movie_texts)[:, 1]
                movie_scores = list(zip(candidate_movie_ids, scores))
                movie_scores.sort(key=lambda x: x[1], reverse=True)
                return movie_scores[:n]
            except Exception as e:
                print(f"CrossEncoder failed: {e}")
                return [(movie_id, random.random()) for movie_id in candidate_movie_ids[:n]]

        else:
            #Lightgbm reranking
            features_df, feature_columns = self.fe.build_inference_features(
                [user_id],
                candidate_movie_ids,
                self.collab_model,
                self.semantic_model
            )

            if features_df.empty:
                return [(movie_id, random.random()) for movie_id in candidate_movie_ids[:n]]

            X = features_df[feature_columns]
            scores = self.ranker.predict_proba(X)[:, 1]

            # Combine movie IDs with scores and sort
            movie_scores = list(zip(features_df['movieId'], scores))
            movie_scores.sort(key=lambda x: x[1], reverse=True)

            return movie_scores[:n]
