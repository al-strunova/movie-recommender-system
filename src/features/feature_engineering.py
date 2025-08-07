import gc
import pickle

import pandas as pd
from src.config import config
from src.data.data_loader import DataLoader


class FeatureEngineering:
    """
    A central class to handle all data processing and feature creation
    for the recommender models.
    """

    def __init__(self):
        """Initializes the class by creating a DataLoader instance."""
        self.loader = DataLoader()
        # Pre-load all dataframes needed to avoid loading them multiple times
        self.ratings_df = self.loader.get_ratings()
        self.movies_df = self.loader.get_movies()
        self.tags_df = self.loader.get_tags()
        self.links_df = self.loader.get_links()
        self.imdb_ratings = self.loader.get_imdb_ratings()
        self.imdb_movies = self.loader.get_imdb_movies()

        # Cache for loaded features
        self._user_features = None
        self._movie_features = None
        self._directors_cache = None
        self._actors_cache = None
        self._quality_cache = None
        self._user_profiles_cache = None

    def _create_training_samples(self):
        """Create training samples with labels."""
        df = self.ratings_df.copy()
        df['label'] = (df['rating'] >= 4).astype(int)
        return df[['userId', 'movieId', 'label']]

    def build_user_genre_profiles(self) -> dict:
        """
        Creates a profile for each user by finding their most frequent genre.
        This is a USER feature.
        """

        merged_df = self.ratings_df.merge(self.movies_df[['movieId', 'genres']], on='movieId')
        liked_movies_df = merged_df[merged_df['rating'] >= 4.0]

        user_profiles_df = (
            liked_movies_df
            .assign(genres_list=lambda df: df['genres'].str.split('|'))
            .explode('genres_list')
            .groupby(['userId', 'genres_list'])
            .size()
            .reset_index(name='genre_count')
            .sort_values('genre_count', ascending=False)
            .drop_duplicates(subset=['userId'], keep='first')
            .set_index('userId')['genres_list']
            .to_dict()
        )
        return user_profiles_df

    def build_enhanced_user_profiles(self) -> dict:
        """
        Creates rich user profiles with multiple genre preferences and behavior patterns.
        Returns dict with user_id -> profile_dict with comprehensive user information.
        """
        merged_df = self.ratings_df.merge(self.movies_df[['movieId', 'genres']], on='movieId')
        liked_movies_df = merged_df[merged_df['rating'] >= 4.0]

        # Get original favorite genre
        user_fav_genres = self.build_user_genre_profiles()

        # Build enhanced profiles
        user_profiles = {}

        for user_id in self.ratings_df['userId'].unique():
            user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
            user_liked = liked_movies_df[liked_movies_df['userId'] == user_id]

            profile = {
                'fav_genre': user_fav_genres.get(user_id)
            }

            # Genre diversity
            if not user_liked.empty:
                all_genres = '|'.join(user_liked['genres']).split('|')
                unique_genres = set(g for g in all_genres if g.strip())
                profile['genre_diversity'] = len(unique_genres)

                # Top 3 genres with counts
                genre_counts = (
                    user_liked
                    .assign(genres_list=lambda df: df['genres'].str.split('|'))
                    .explode('genres_list')
                    .groupby('genres_list')
                    .size()
                    .sort_values(ascending=False)
                    .head(3)
                )
                profile['top_genres'] = genre_counts.to_dict()
            else:
                profile['genre_diversity'] = 0
                profile['top_genres'] = {}

            user_profiles[user_id] = profile

        print(f"Built enhanced profiles for {len(user_profiles)} users")
        return user_profiles

    def _get_personnel(self, personnel_type: str, n_per_movie: int = 1, semantic_format: bool = False) -> pd.DataFrame:
        """
        A generic and robust helper to get top N directors or actors for each movie.
        """
        # Load data as needed, relying on DataLoader's caching
        links_df = self.loader.get_links()
        names_df = self.loader.get_imdb_names()

        if personnel_type == 'director':
            personnel_df = self.loader.get_imdb_crew()
            personnel_df = (
                personnel_df.dropna(subset=['directors'])
                .assign(nconst=lambda df: df['directors'].str.split(','))
                .explode('nconst')
            )
        else:  # 'actor' or 'actress'
            personnel_df = self.loader.get_imdb_principals()
            # Filter for actors/actresses and top N by ordering
            personnel_df = personnel_df[personnel_df['ordering'] <= n_per_movie]

        personnel_with_names = (
            links_df[['movieId', 'tconst']]
            .merge(personnel_df[['tconst', 'nconst']], on='tconst', how='inner')
            .merge(names_df[['nconst', 'primaryName']], on='nconst', how='inner')
        )

        if semantic_format:
            personnel_with_names['name_clean'] = personnel_with_names['primaryName'].str.lower()
            separator = ', '
        else:
            squashed_names = personnel_with_names['primaryName'].str.lower().str.replace(' ', '', regex=False)
            personnel_with_names['name_clean'] = f'{personnel_type}_' + squashed_names
            separator = ' '

        return (
            personnel_with_names
            .groupby('movieId')['name_clean']
            .agg(separator.join)
            .reset_index()
            .rename(columns={'name_clean': f'{personnel_type}s'})
        )

    def _get_user_favorite_directors(self, user_id: int, limit: int = 3) -> str:
        """Get user's favorite directors from cached data."""
        # Get user's highly rated movies
        user_liked = self.ratings_df[
            (self.ratings_df['userId'] == user_id) &
            (self.ratings_df['rating'] >= 4)
            ]['movieId'].tolist()

        directors_count = {}
        for movie_id in user_liked:
            directors_row = self._directors_cache[self._directors_cache['movieId'] == movie_id]
            if not directors_row.empty and directors_row.iloc[0]['directors']:
                directors = directors_row.iloc[0]['directors']
                for director in directors.split(', '):
                    directors_count[director] = directors_count.get(director, 0) + 1

        # Return top directors
        top_directors = sorted(directors_count.items(), key=lambda x: x[1], reverse=True)[:limit]
        return ', '.join([director for director, _ in top_directors])

    def _get_user_favorite_actors(self, user_id: int, limit: int = 5) -> str:
        """Get user's favorite actors from cached data."""
        # Get user's highly rated movies
        user_liked = self.ratings_df[
            (self.ratings_df['userId'] == user_id) &
            (self.ratings_df['rating'] >= 4)
            ]['movieId'].tolist()

        actors_count = {}
        for movie_id in user_liked:
            actors_row = self._actors_cache[self._actors_cache['movieId'] == movie_id]
            if not actors_row.empty and actors_row.iloc[0]['actors']:
                actors = actors_row.iloc[0]['actors']
                for actor in actors.split(', '):
                    actors_count[actor] = actors_count.get(actor, 0) + 1

        # Return top actors
        top_actors = sorted(actors_count.items(), key=lambda x: x[1], reverse=True)[:limit]
        return ', '.join([actor for actor, _ in top_actors])

    def build_movie_documents(self, semantic_format: bool = False) -> pd.Series:
        """
        The main public method that orchestrates the feature building.
        Args:
            semantic_format: If True, creates natural text for SentenceTransformers.
                             If False (default), creates tokenized text for TF-IDF.
        """
        movies_df = self.loader.get_movies()

        # Get movie genre
        movie_docs = movies_df[['movieId', 'genres']].copy()
        movie_docs['genres'] = movie_docs['genres'].str.replace('|', ' ', regex=False).str.lower()

        # Get all movie tags using helper method
        all_movie_tags = {movie_id: self._get_all_movie_tags(movie_id)
                          for movie_id in movies_df['movieId']}
        movie_docs['tag'] = movie_docs['movieId'].map(all_movie_tags).fillna('')

        if not semantic_format:
            # TF-IDF: Replace spaces with underscores in tags
            movie_docs['tag'] = movie_docs['tag'].str.replace(' ', '_', regex=False)

        # Get aggregated directors and top 3 actors using the corrected helper
        directors = self._get_personnel('director', 1, semantic_format)
        actors = self._get_personnel('actor', 2, semantic_format)

        # Merge all content features into the movie_docs DataFrame
        movie_docs = movie_docs.merge(directors, on='movieId', how='left')
        movie_docs = movie_docs.merge(actors, on='movieId', how='left')

        # 5. Fill any missing data and combine into the final document string
        movie_docs.fillna('', inplace=True)

        # Combine features
        documents = (
                movie_docs['genres'] + ' ' +
                movie_docs['tag'] + ' ' +
                movie_docs['directors'] + ' ' +
                movie_docs['actors']
        ).str.strip().str.replace(r'\s+', ' ', regex=True)

        documents.index = movie_docs['movieId']

        print("Built movie documents.")
        return documents

    def build_scored_movies(self, m_quantile: float = None) -> tuple[pd.DataFrame, float, float]:
        """
        Creates the master movie DataFrame, including the calculated
        IMDb weighted rating (popularity score).
        """
        imdb_ratings = self.imdb_ratings.copy()

        # Ensure numeric types
        imdb_ratings['numVotes'] = pd.to_numeric(imdb_ratings['numVotes'], errors='coerce')
        imdb_ratings['averageRating'] = pd.to_numeric(imdb_ratings['averageRating'], errors='coerce')
        imdb_ratings.dropna(subset=['averageRating', 'numVotes'], inplace=True)

        # Perform all the merges
        movie_ratings = pd.merge(self.links_df, imdb_ratings, on='tconst', how='inner')
        movie_ratings = pd.merge(self.movies_df, movie_ratings, on='movieId', how='inner')

        # Calculate weighted rating (as before)
        if m_quantile is None:
            m_quantile = config.params.recommender.popularity_m_quantile
        C = movie_ratings['averageRating'].mean()
        m = movie_ratings['numVotes'].quantile(m_quantile)
        R = movie_ratings['averageRating']
        v = movie_ratings['numVotes']
        movie_ratings['weighted_rating'] = (v / (v + m)) * R + (m / (v + m)) * C

        final_cols = ['movieId', 'averageRating', 'numVotes', 'weighted_rating']
        final_df = movie_ratings.sort_values('weighted_rating', ascending=False)[final_cols].reset_index(drop=True)

        return final_df, m, C

    def build_top_movies_per_genre_cache(self, m_quantile: float = None) -> dict:
        """
        Creates a cache of pre-ranked (movieId, score) tuples for each genre.
        This is a key data product for the GenrePopularityRecommender.
        """
        # This method needs the scored movies first
        scored_movies_df, _, _ = self.build_scored_movies(m_quantile)
        movies_with_scores_and_genres = pd.merge(scored_movies_df,
                                                 self.movies_df[['movieId', 'genres']],
                                                 on='movieId', how='left')

        exploded_genres_df = (
            movies_with_scores_and_genres
            .assign(genre=lambda df: df['genres'].str.split('|'))
            .explode('genre')
        )
        sorted_movies = exploded_genres_df.sort_values('weighted_rating', ascending=False)

        return (
            sorted_movies
            .groupby('genre')[['movieId', 'weighted_rating']]
            .apply(lambda g: list(zip(g['movieId'], g['weighted_rating'])))
            .to_dict()
        )

    def build_user_feature_set(self) -> pd.DataFrame:
        """Build user features and save them."""
        print("Building user features...")

        # Basic stats
        user_stats = self.ratings_df.groupby('userId')['rating'].agg(['mean', 'count', 'std']).reset_index()
        user_stats.columns = ['userId', 'user_avg_rating', 'user_rating_count', 'user_rating_stddev']

        # Favorite genre
        user_fav_genre_map = self.build_user_genre_profiles()

        # Genre diversity
        merged_df = self.ratings_df.merge(self.movies_df[['movieId', 'genres']], on='movieId')
        liked_movies_df = merged_df[merged_df['rating'] >= 4.0]
        user_genre_diversity = (
            liked_movies_df
            .groupby('userId')['genres']
            .apply(lambda x: len(set('|'.join(x).split('|'))))
            .rename('user_genre_diversity')
            .reset_index()
        )

        # Combine
        final_user_features = user_stats
        final_user_features['user_fav_genre'] = final_user_features['userId'].map(user_fav_genre_map)
        final_user_features = final_user_features.merge(user_genre_diversity, on='userId', how='left')
        final_user_features.fillna(0, inplace=True)

        final_user_features = final_user_features.fillna({
            'user_rating_stddev': 0,
            'user_genre_diversity': 1
        })

        with open(config.paths.USER_FEATURES_PATH, 'wb') as f:
            pickle.dump(final_user_features, f)
        print(f" User features saved")
        return final_user_features

    def build_movie_feature_set(self) -> pd.DataFrame:
        """Build movie features and save them."""
        print("Building movie features...")

        final_movie_df = pd.merge(self.movies_df[['movieId', 'genres']],
                                  self.links_df[['movieId', 'tconst']],
                                  on='movieId')

        # Add movies scores
        movies_with_scores_df, _, _ = self.build_scored_movies()
        final_movie_df = final_movie_df.merge(movies_with_scores_df, on='movieId', how='left')

        # Add movie extra information
        final_movie_df = final_movie_df.merge(
            self.imdb_movies[['tconst', 'isAdult', 'startYear', 'runtimeMinutes']], on='tconst', how='left')
        for col in ['startYear', 'runtimeMinutes', 'isAdult']:
            final_movie_df[col] = pd.to_numeric(final_movie_df[col], errors='coerce')

        # Add tag-based features
        movie_tag_counts = self.tags_df.groupby('movieId').size()
        final_movie_df['movie_tag_count'] = final_movie_df['movieId'].map(movie_tag_counts).fillna(0).astype(int)
        final_movie_df['movie_has_tags'] = (final_movie_df['movie_tag_count'] > 0).astype(int)

        # Movie popularity (how many ratings)
        movie_popularity = self.ratings_df.groupby('movieId').size()
        final_movie_df['movie_popularity'] = final_movie_df['movieId'].map(movie_popularity)

        # Item Features based on top movie genre
        final_movie_df['movie_genre_count'] = final_movie_df['genres'].str.split('|').str.len()
        final_movie_df['is_single_genre'] = (final_movie_df['movie_genre_count'] == 1).astype(int)
        top_genres = ['Drama', 'Comedy', 'Action', 'Romance', 'Thriller']
        for genre in top_genres:
            final_movie_df[f'has_genre_{genre.lower()}'] = final_movie_df['genres'].str.contains(genre).astype(int)

        # Fill missing values
        final_movie_df = final_movie_df.fillna({
            'averageRating': final_movie_df['averageRating'].mean(),
            'numVotes': 0, 'startYear': 2000,
            'weighted_rating': final_movie_df['weighted_rating'].mean(),
            'runtimeMinutes': 90, 'isAdult': 0, 'movie_popularity': 1, 'movie_tag_count': 0
        })

        # Save
        final_movie_df = final_movie_df.drop(['tconst'], axis=1, errors='ignore')
        with open(config.paths.MOVIE_FEATURES_PATH, 'wb') as f:
            pickle.dump(final_movie_df, f)
        print(f"Movie features saved")
        return final_movie_df

    def _ensure_features_exist(self):
        """Ensure features exist, build if they don't."""
        if not config.paths.USER_FEATURES_PATH.exists():
            self.build_user_feature_set()
        if not config.paths.MOVIE_FEATURES_PATH.exists():
            self.build_movie_feature_set()

    def _load_features(self):
        """Load features once and cache them."""
        if self._user_features is None:
            with open(config.paths.USER_FEATURES_PATH, 'rb') as f:
                self._user_features = pickle.load(f)
        if self._movie_features is None:
            with open(config.paths.MOVIE_FEATURES_PATH, 'rb') as f:
                self._movie_features = pickle.load(f)

    def get_candidate_scores(self, user_movie_pairs: pd.DataFrame, collab_model=None,
                             semantic_model=None) -> pd.DataFrame:
        """Get scores from candidate models."""

        from src.recommenders.collaborative_model import CollaborativeFilteringModel
        from src.recommenders.semantic_recommender import SemanticRecommender

        pairs = user_movie_pairs.copy()

        # Track if we loaded models ourselves
        loaded_collab = collab_model is None
        loaded_semantic = semantic_model is None

        # Load if not provided
        if collab_model is None:
            try:
                collab_model = CollaborativeFilteringModel.load()
            except:
                collab_model = None

        if semantic_model is None:
            try:
                semantic_model = SemanticRecommender.load()
            except:
                semantic_model = None

        # Use models...
        if collab_model:
            pairs['collab_score'] = pairs.apply(
                lambda row: collab_model.predict_score(row['userId'], row['movieId']), axis=1
            )
        else:
            pairs['collab_score'] = 0.0

        if semantic_model:
            pairs['content_sim_score'] = pairs.apply(
                lambda row: semantic_model.predict_score(row['userId'], row['movieId']), axis=1
            )
        else:
            pairs['content_sim_score'] = 0.0

        # Cleanup only models we loaded ourselves
        if loaded_collab and collab_model:
            collab_model.cleanup()
            del collab_model
        if loaded_semantic and semantic_model:
            semantic_model.cleanup()
            del semantic_model

        if loaded_collab or loaded_semantic:
            gc.collect()

        return pairs[['userId', 'movieId', 'collab_score', 'content_sim_score']]

    def build_ranking_dataset(self) -> tuple:
        """Build complete ranking dataset for training."""
        print("Building dataset for a ranking model...")

        # Create user-movie pairs with labels
        training_df = self._create_training_samples()

        # Build the feature tables
        user_features = self.build_user_feature_set()
        movie_features = self.build_movie_feature_set()

        final_df = training_df.merge(user_features, on='userId')
        final_df = final_df.merge(movie_features, on='movieId')

        # Features combined with user and item features
        final_df['has_user_fav_genre'] = final_df.apply(
            lambda row: 1 if pd.notna(row['user_fav_genre']) and str(row['user_fav_genre']) in str(
                row['genres']) else 0,
            axis=1
        )
        user_tagged = self.tags_df[['userId', 'movieId']].drop_duplicates()
        user_tagged['user_tagged_movie'] = 1
        final_df = final_df.merge(user_tagged, on=['userId', 'movieId'], how='left')
        final_df['user_tagged_movie'] = final_df['user_tagged_movie'].fillna(0).astype(int)

        # Add candidate model scores
        candidate_scores = self.get_candidate_scores(final_df[['userId', 'movieId']])
        final_df = final_df.merge(candidate_scores, on=['userId', 'movieId'])

        # At the end, select features and handle missing values
        feature_cols = [
            # Candidate scores
            'collab_score', 'content_sim_score',

            # User features
            'user_avg_rating', 'user_rating_count', 'user_rating_stddev',
            'user_genre_diversity', 'has_user_fav_genre',

            # Movie features
            'movie_popularity', 'averageRating', 'numVotes', 'weighted_rating',
            'startYear', 'runtimeMinutes', 'movie_genre_count', 'isAdult', 'is_single_genre',

            # Tag features
            'movie_tag_count', 'user_tagged_movie',

            # Genre flags
            *[f'has_genre_{g.lower()}' for g in ['Drama', 'Comedy', 'Action', 'Romance', 'Thriller']]
        ]

        # Prepare final output
        X = final_df[feature_cols]
        y = final_df['label']
        metadata = final_df[['userId', 'movieId']]

        return X, y, metadata

    def build_inference_features(self, user_ids: list, movie_ids: list, collab_model=None,
                                 semantic_model=None) -> tuple:
        """Build features for inference - uses saved features!"""

        # Load cached features
        self._ensure_features_exist()
        self._load_features()

        # Create all user-movie pairs
        pairs = []
        for user_id in user_ids:
            for movie_id in movie_ids:
                pairs.append({'userId': user_id, 'movieId': movie_id})
        pairs_df = pd.DataFrame(pairs)

        user_data = self._user_features[self._user_features['userId'].isin(user_ids)]

        # Handle missing users
        missing_users = set(user_ids) - set(user_data['userId'])
        if missing_users:
            default_user_data = []
            for user_id in missing_users:
                default_user_data.append({
                    'userId': user_id, 'user_avg_rating': 3.5, 'user_rating_count': 0,
                    'user_rating_stddev': 0, 'user_genre_diversity': 1, 'user_fav_genre': None
                })
            default_df = pd.DataFrame(default_user_data)
            user_data = pd.concat([user_data, default_df], ignore_index=True)

        # Get movie data
        movie_data = self._movie_features[self._movie_features['movieId'].isin(movie_ids)]

        # Handle missing movies
        missing_movies = set(movie_ids) - set(movie_data['movieId'])
        if missing_movies:
            print(f"Warning: {len(missing_movies)} movies not found in features: {list(missing_movies)[:5]}...")
            # Remove pairs with missing movies
            pairs_df = pairs_df[pairs_df['movieId'].isin(movie_data['movieId'])]

        if pairs_df.empty:
            print("No valid user-movie pairs found!")
            return pd.DataFrame(), []

        # Merge features
        final_df = pairs_df.merge(user_data, on='userId', suffixes=('', '_user'))
        final_df = final_df.merge(movie_data, on='movieId', suffixes=('', '_movie'))

        # Add interaction features
        final_df['has_user_fav_genre'] = final_df.apply(
            lambda row: 1 if pd.notna(row['user_fav_genre']) and
                             str(row['user_fav_genre']) in str(row['genres']) else 0, axis=1
        )

        # Add tagging
        user_tagged = self.tags_df[
            (self.tags_df['userId'].isin(user_ids)) &
            (self.tags_df['movieId'].isin(movie_ids))
            ][['userId', 'movieId']].drop_duplicates()
        user_tagged['user_tagged_movie'] = 1
        final_df = final_df.merge(user_tagged, on=['userId', 'movieId'], how='left')
        final_df['user_tagged_movie'] = final_df['user_tagged_movie'].fillna(0).astype(int)

        # Get candidate scores for these pairs
        candidate_scores = self.get_candidate_scores(final_df[['userId', 'movieId']], collab_model, semantic_model)
        final_df = final_df.merge(candidate_scores, on=['userId', 'movieId'])

        # Same features as training
        feature_cols = [
            'collab_score', 'content_sim_score',
            'user_avg_rating', 'user_rating_count', 'user_rating_stddev',
            'user_genre_diversity', 'has_user_fav_genre',
            'movie_popularity', 'averageRating', 'numVotes', 'weighted_rating',
            'startYear', 'runtimeMinutes', 'movie_genre_count', 'isAdult', 'is_single_genre',
            'movie_tag_count', 'user_tagged_movie',
            'has_genre_drama', 'has_genre_comedy', 'has_genre_action',
            'has_genre_romance', 'has_genre_thriller'
        ]

        return final_df[['userId', 'movieId'] + feature_cols], feature_cols

    def create_user_text_representation(self, user_id: int, user_profile: dict) -> str:
        """Create focused user representation emphasizing preferences."""
        parts = []

        # Genre preferences (most important)
        fav_genre = user_profile.get('fav_genre')
        if fav_genre:
            parts.append(f"Prefers {fav_genre}")

        # Genre diversity
        genre_diversity = user_profile.get('genre_diversity', 0)
        if genre_diversity > 5:
            parts.append("enjoys many genres")
        elif genre_diversity <= 2:
            parts.append("focused on specific genres")

        # Top genres
        top_genres = user_profile.get('top_genres', {})
        if len(top_genres) > 1:
            genre_names = list(top_genres.keys())[:3]
            parts.append(f"also likes {', '.join(genre_names)}")

        # User tags (most semantic signal)
        user_tags = self._get_all_user_tags(user_id)
        if user_tags:
            parts.append(f"Interests: {user_tags}")

        # Favorite directors/actors
        fav_directors = self._get_user_favorite_directors(user_id)
        if fav_directors:
            parts.append(f"likes directors: {fav_directors}")

        fav_actors = self._get_user_favorite_actors(user_id)
        if fav_actors:
            parts.append(f"likes actors: {fav_actors}")

        return ". ".join(parts) if parts else "General movie viewer"

    def create_movie_text_representation(self, movie_id: int) -> str:
        """Create natural language movie representation for CrossEncoder."""
        parts = []

        # Get movie basic info from movies_df
        movie_row = self.movies_df[self.movies_df['movieId'] == movie_id]
        if movie_row.empty:
            return ""

        genres = movie_row.iloc[0]['genres']
        if genres:
            clean_genres = genres.replace('|', ', ')
            parts.append(clean_genres)

        # Directors and actors
        try:
            directors_df = self._directors_cache
            directors_row = directors_df[directors_df['movieId'] == movie_id]
            if not directors_row.empty and directors_row.iloc[0]['directors']:
                parts.append(f"directed by {directors_row.iloc[0]['directors']}")
        except:
            pass

        try:
            actors_df = self._actors_cache
            actors_row = actors_df[actors_df['movieId'] == movie_id]
            if not actors_row.empty and actors_row.iloc[0]['actors']:
                parts.append(f"starring {actors_row.iloc[0]['actors']}")
        except:
            pass

        # Quality indicators (highly rated, popular, etc.)
        try:
            quality_df = self._quality_cache
            quality_row = quality_df[quality_df['movieId'] == movie_id]
            if not quality_row.empty and quality_row.iloc[0]['quality_indicators'].strip():
                parts.append(quality_row.iloc[0]['quality_indicators'])
        except:
            pass

        # All movie tags (using the helper method)
        movie_tags = self._get_all_movie_tags(movie_id)
        if movie_tags:
            parts.append(f"tags: {movie_tags}")

        return ". ".join(parts)

    def _get_movie_quality_indicators(self) -> pd.DataFrame:
        """
        Get movie quality/popularity as text indicators for CrossEncoder.
        Converts numerical ratings/votes to descriptive text like "highly rated", "popular".
        """

        # Get movie scores and IMDb data
        scored_movies_df, _, _ = self.build_scored_movies()

        # Create quality indicators as text
        quality_indicators = []

        for _, row in scored_movies_df.iterrows():
            indicators = []

            # Rating quality
            rating = row.get('averageRating')
            if pd.notna(rating) and rating > 0:
                if rating >= 8.0:
                    indicators.append("highly rated")
                elif rating >= 7.0:
                    indicators.append("well rated")
                elif rating <= 5.0:
                    indicators.append("low rated")

            # Popularity based on vote count
            votes = row.get('numVotes', 0)
            if votes > 100000:
                indicators.append("very popular")
            elif votes > 25000:
                indicators.append("popular")
            elif votes > 5000:
                indicators.append("known")

            quality_indicators.append(' '.join(indicators))

        scored_movies_df['quality_indicators'] = quality_indicators
        return scored_movies_df[['movieId', 'quality_indicators']]

    def _ensure_movie_cache_loaded(self):
        """Load movie data cache if not already loaded."""
        if self._directors_cache is None:
            self._directors_cache = self._get_personnel('director', 1, semantic_format=True)

        if self._actors_cache is None:
            self._actors_cache = self._get_personnel('actor', 3, semantic_format=True)

        if self._quality_cache is None:
            self._quality_cache = self._get_movie_quality_indicators()

        if self._user_profiles_cache is None:
            self._user_profiles_cache = self.build_enhanced_user_profiles()

    def build_crossencoder_inference_data(self, user_ids: list, movie_ids: list):
        """
        Build text representations for CrossEncoder inference.

        Args:
            user_ids: List of user IDs
            movie_ids: List of movie IDs

        Returns:
            user_texts: List of user text representations for all user-movie pairs
            movie_texts: List of movie text representations for all user-movie pairs
        """
        # Ensure movie cache is loaded
        self._ensure_movie_cache_loaded()

        user_texts = []
        movie_texts = []

        for user_id in user_ids:
            # Get user profile and create text representation
            profile = self._user_profiles_cache.get(user_id, {})
            user_text = self.create_user_text_representation(user_id, profile)

            for movie_id in movie_ids:
                # Create movie text representation
                movie_text = self.create_movie_text_representation(movie_id)

                user_texts.append(user_text)
                movie_texts.append(movie_text)

        return user_texts, movie_texts

    # Helper methods (already defined in your request):
    def _get_all_user_tags(self, user_id: int) -> str:
        """Get all unique tags used by a user."""
        user_tags = self.tags_df[self.tags_df['userId'] == user_id]['tag'].unique()
        return ', '.join(user_tags) if len(user_tags) > 0 else ''

    def _get_all_movie_tags(self, movie_id: int) -> str:
        """Get all unique tags for a movie."""
        movie_tags = self.tags_df[self.tags_df['movieId'] == movie_id]['tag'].unique()
        return ', '.join(movie_tags) if len(movie_tags) > 0 else ''
