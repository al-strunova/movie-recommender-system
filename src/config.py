from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class PathsConfig:
    """A dataclass to hold all project paths."""
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    SRC_DIR: Path = PROJECT_ROOT / "src"
    CANDIDATE_MODEL_DIR: Path = MODELS_DIR / "candidate_models"
    RANKING_MODEL_DIR: Path = MODELS_DIR / "ranking_models"
    SRC_DATA_FEATURE_DIR: Path = DATA_DIR / "features"

    # Data
    MOVIELENS_DIR: Path = DATA_DIR / "movielens"
    IMDB_DIR: Path = DATA_DIR / "imdb"

    RATINGS_FILE: Path = MOVIELENS_DIR / "ratings.csv"
    MOVIES_FILE: Path = MOVIELENS_DIR / "movies.csv"
    LINKS_FILE: Path = MOVIELENS_DIR / "links.csv"
    TAGS_FILE: Path = MOVIELENS_DIR / "tags.csv"

    IMDB_RATINGS_FILE: Path = IMDB_DIR / "title.ratings.tsv.gz"
    IMDB_CREW_FILE: Path = IMDB_DIR / "title.crew.tsv.gz"
    IMDB_PRINCIPALS_FILE: Path = IMDB_DIR / "title.principals.tsv.gz"
    IMDB_NAMES_FILE: Path = IMDB_DIR / "name.basics.tsv.gz"
    IMDB_MOVIES_FILE: Path = IMDB_DIR / "title.basics.tsv.gz"

    # Saved Model Paths
    POPULARITY_MODEL_PATH: Path = CANDIDATE_MODEL_DIR / "popularity_model.pkl"
    GENRE_MODEL_PATH: Path = CANDIDATE_MODEL_DIR / "genre_model.pkl"
    ALS_MODEL_PATH: Path = CANDIDATE_MODEL_DIR / "als_model.npz"
    ALS_MAPS_PATH: Path = CANDIDATE_MODEL_DIR / "als_maps.pkl"
    ALS_MATRIX_PATH: Path = CANDIDATE_MODEL_DIR / "als_user_item_matrix.npz"
    LIGHTGBM_RANKER_PATH: Path = RANKING_MODEL_DIR / "lightgbm_ranker.pkl"
    CROSS_ENCODER_RANKER_PATH: Path = RANKING_MODEL_DIR / "cross_encoder_ranker"
    GNN_MODEL_PATH: Path = CANDIDATE_MODEL_DIR / "gnn_recommender.pt"

    # Feature storage
    USER_FEATURES_PATH = SRC_DATA_FEATURE_DIR / "user_features.pkl"
    MOVIE_FEATURES_PATH = SRC_DATA_FEATURE_DIR / "movie_features.pkl"


@dataclass
class ALSParams:
    """Hyperparameters for the Collaborative Filtering model."""
    factors: int = 50
    regularization: float = 0.01
    iterations: int = 20


@dataclass
class ContentParams:
    """Hyperparameters for Content-Based models."""
    svd_components: int = 100
    tfidf_max_features: int = 20000
    sentence_transformer_model: str = 'all-MiniLM-L6-v2'
    sentence_transformer_batch_size: int = 64


@dataclass
class RecommenderParams:
    """Hyperparameters for recommendation and ranking logic."""
    n_seed_movies: int = 5
    n_candidates_per_movie: int = 50
    n_final_recommendations: int = 10
    similarity_score_weight: float = 0.2
    popularity_score_weight: float = 0.8
    popularity_m_quantile: float = 0.70

@dataclass
class LightGBMParams:
    """Parameters for LightGBM ranker."""
    lgb_params: dict = field(default_factory=lambda: {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 100,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1
    })
    test_size = 0.2
    random_state = 42

@dataclass
class GNNParams:
    """Parameters for GNN recommender."""
    hidden_dim: int = 256
    output_dim: int = 128
    num_epochs: int = 200
    learning_rate: float = 0.001
    dropout: float = 0.2
    rating_threshold: float = 3.5

@dataclass
class CrossEncoderParams:
    """Parameters for CrossEncoder ranker."""
    model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'


@dataclass
class TwoStageParams:
    n_candidates = 50


@dataclass
class EvaluationParams:
    """Parameters for model evaluation."""
    k_for_metrics: int = 10
    test_set_size: float = 0.2
    random_seed: int = 42


@dataclass
class ParamsConfig:
    """A dataclass to hold all model hyperparameters."""
    als: ALSParams = field(default_factory=ALSParams)
    content: ContentParams = field(default_factory=ContentParams)
    recommender: RecommenderParams = field(default_factory=RecommenderParams)
    evaluation: EvaluationParams = field(default_factory=EvaluationParams)
    lgb_ranking: LightGBMParams = field(default_factory=LightGBMParams)
    two_stage: TwoStageParams = field(default_factory=TwoStageParams)
    cross_encoder: CrossEncoderParams = field(default_factory=CrossEncoderParams)
    gnn: GNNParams = field(default_factory=GNNParams)


@dataclass
class Config:
    """The main, centralized configuration object for the entire project."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    params: ParamsConfig = field(default_factory=ParamsConfig)


config = Config()
