import pandas as pd
from sklearn.metrics import ndcg_score

from src.config import config
from src.recommenders.gnn_recommender import GNNRecommender
from src.recommenders.popularity_model import PopularityModel
from src.recommenders.genre_popularity_recommender import GenrePopularityRecommender
from src.recommenders.collaborative_model import CollaborativeFilteringModel
from src.recommenders.hybrid_recommender import HybridRecommender
from src.recommenders.content_based_recommender import ContentBasedRecommender
from src.recommenders.semantic_recommender import SemanticRecommender
from src.recommenders.two_stage_recommender import TwoStageRecommender


def load_and_prepare_data():
    """Loads and prepares data for evaluation."""
    print("=" * 50)
    print("Loading user ratings and preparing evaluation data...")

    ratings_df = pd.read_csv(config.paths.RATINGS_FILE)
    movies_df = pd.read_csv(config.paths.MOVIES_FILE)

    relevance_signal = ratings_df['rating'].apply(
        lambda r: 1 if r >= 4 else (0 if r <= 2 else None)
    )
    strong_signal_df = ratings_df[relevance_signal.notna()].copy()
    strong_signal_df['relevant'] = relevance_signal[relevance_signal.notna()].astype(int)

    test_df = strong_signal_df.groupby('userId').sample(frac=config.params.evaluation.test_set_size,
                                                        random_state=config.params.evaluation.random_seed)
    train_df = strong_signal_df.drop(test_df.index)

    ground_truth = test_df[test_df['relevant'] == 1].groupby('userId')['movieId'].apply(set).to_dict()
    seen_movies = train_df.groupby('userId')['movieId'].apply(set).to_dict()
    catalog_ids = set(movies_df['movieId'].unique())

    return ground_truth, seen_movies, catalog_ids


def calculate_metrics(recommended_ids, actual_likes, k):
    """Calculates precision, recall, and nDCG for a single user."""
    hits = len(recommended_ids.intersection(actual_likes))

    precision = hits / k if k > 0 else 0
    recall = hits / len(actual_likes) if len(actual_likes) > 0 else 0

    # Create the y_true list based on the full ranked list of recommendations
    y_true = [1 if rec_id in actual_likes else 0 for rec_id in recommended_ids]
    y_score = [k - i for i in range(len(y_true))]  # Use len(y_true) in case fewer than k recs were returned

    ndcg = ndcg_score([y_true], [y_score], k=k) if hits > 0 else 0

    return {'precision': precision, 'recall': recall, 'ndcg': ndcg}


def run_evaluation(model, recommend_method_name, ground_truth, seen_movies, catalog_ids,
                   k: int = config.params.evaluation.k_for_metrics):
    """Runs the full evaluation loop for a given model and method."""
    print("\n" + "=" * 50)
    print(f"EVALUATING MODEL: {model.__class__.__name__}")

    # Add description if available
    if hasattr(model, 'get_candidates_description'):
        print(f"Description: {model.get_candidates_description()}")

    print(f"Testing Method: {recommend_method_name}")
    print("=" * 50)

    all_scores = []
    all_recommended_items = set()
    test_users = list(ground_truth.keys())

    # Get the specific recommendation function from the model object
    recommend_function = getattr(model, recommend_method_name)

    for user_id in test_users:
        actual_likes = ground_truth.get(user_id, set())
        if not actual_likes:
            continue

        seen = seen_movies.get(user_id, set())
        recommendations = recommend_function(user_id=user_id, n=k, exclude_seen=seen)

        if not recommendations:
            continue

        recommended_ids = {rec[0] for rec in recommendations}
        all_recommended_items.update(recommended_ids)

        user_scores = calculate_metrics(recommended_ids, actual_likes, k)
        all_scores.append(user_scores)

    results_df = pd.DataFrame(all_scores)
    avg_results = results_df.mean()
    coverage = len(all_recommended_items) / len(catalog_ids)

    # --- Print Final Report Card ---
    print("\n--- Overall Metrics ---")
    print(f"Precision@{k}: {avg_results['precision']:.4f}")
    print(f"Recall@{k}:    {avg_results['recall']:.4f}")
    print(f"nDCG@{k}:      {avg_results['ndcg']:.4f}")
    print(f"Coverage:      {coverage:.2%}")
    print("-" * 25)

    return avg_results, coverage


if __name__ == "__main__":

    # Prepare data once for all models
    ground_truth_data, seen_movies_data, catalog_ids = load_and_prepare_data()

    # Load all models once
    print("\n" + "#" * 50)
    print("FITTING ALL MODELS")
    print("#" * 50)

    popularity_model = PopularityModel.load()
    genre_model = GenrePopularityRecommender.load()
    collab_model = CollaborativeFilteringModel.load()
    content_based_model = ContentBasedRecommender.load()
    semantic_model = SemanticRecommender.load()
    hybrid_model = HybridRecommender(
        collaborative_model=collab_model,
        popularity_model=popularity_model
    )
    gnn_model = GNNRecommender.load()

    # Define all models and methods to test
    models_to_test = {
        # Regular candidate models
        'popularity_static': (popularity_model, 'get_top_recommendations'),
        'popularity_weighted': (popularity_model, 'get_top_weighted_recommendations'),
        'genre_popularity': (genre_model, 'recommend'),
        'collaborative': (collab_model, 'recommend'),
        'hybrid': (hybrid_model, 'recommend'),
        'content_similarity': (content_based_model, 'recommend_by_similarity'),
        'content_hybrid': (content_based_model, 'recommend_by_hybrid_score'),
        'semantic_similarity': (semantic_model, 'recommend_by_similarity'),
        'semantic_hybrid': (semantic_model, 'recommend_by_hybrid_score'),
        'gnn': (gnn_model, 'recommend'),

        # Two-stage models with different candidate combinations
        'two_stage_lgb_default': (
            TwoStageRecommender(
                candidate_models=[(collab_model, 'recommend'),(semantic_model, 'recommend_by_hybrid_score')],
                collab_model=collab_model,
                semantic_model=semantic_model,
                use_cross_encoder=False
            ), 'recommend'
        ),
        'two_stage_lgb_popularity_static': (
            TwoStageRecommender(
                candidate_models=[(popularity_model, 'get_top_recommendations')],
                collab_model=collab_model,
                semantic_model=semantic_model,
                use_cross_encoder=False
            ), 'recommend'
        ),
        'two_stage_lgb_popularity_weighted': (
            TwoStageRecommender(
                candidate_models=[(popularity_model, 'get_top_weighted_recommendations')],
                collab_model=collab_model,
                semantic_model=semantic_model,
                use_cross_encoder=False
            ), 'recommend'
        ),
        'two_stage_lgb_genre_popularity': (
            TwoStageRecommender(
                candidate_models=[(genre_model, 'recommend')],
                collab_model=collab_model,
                semantic_model=semantic_model,
                use_cross_encoder=False
            ), 'recommend'
        ),
        'two_stage_lgb_collaborative': (
            TwoStageRecommender(
                candidate_models=[(collab_model, 'recommend')],
                collab_model=collab_model,
                semantic_model=semantic_model,
                use_cross_encoder=False
            ), 'recommend'
        ),
        'two_stage_lgb_hybrid': (
            TwoStageRecommender(
                candidate_models=[(hybrid_model, 'recommend')],
                collab_model=collab_model,
                semantic_model=semantic_model,
                use_cross_encoder=False
            ), 'recommend'
        ),
        'two_stage_lgb_content_similarity': (
            TwoStageRecommender(
                candidate_models=[(content_based_model, 'recommend_by_similarity')],
                collab_model=collab_model,
                semantic_model=semantic_model,
                use_cross_encoder=False
            ), 'recommend'
        ),
        'two_stage_lgb_content_hybrid': (
            TwoStageRecommender(
                candidate_models=[(content_based_model, 'recommend_by_hybrid_score')],
                collab_model=collab_model,
                semantic_model=semantic_model,
                use_cross_encoder=False
            ), 'recommend'
        ),
        'two_stage_lgb_semantic_similarity': (
            TwoStageRecommender(
                candidate_models=[(semantic_model, 'recommend_by_similarity')],
                collab_model=collab_model,
                semantic_model=semantic_model,
                use_cross_encoder=False
            ), 'recommend'
        ),
        'two_stage_lgb_semantic_hybrid': (
            TwoStageRecommender(
                candidate_models=[(semantic_model, 'recommend_by_hybrid_score')],
                collab_model=collab_model,
                semantic_model=semantic_model,
                use_cross_encoder=False
            ), 'recommend'
        ),
        'two_stage_lgb_gnn': (
            TwoStageRecommender(
                candidate_models=[(gnn_model, 'recommend')],
                collab_model=collab_model,
                semantic_model=semantic_model,
                use_cross_encoder=False
            ), 'recommend'
        ),
        'two_stage_lgb_all_models': (
            TwoStageRecommender(
                candidate_models=[
                    (popularity_model, 'get_top_recommendations'),
                    (collab_model, 'recommend'),
                    (semantic_model, 'recommend_by_hybrid_score')],
                collab_model=collab_model,
                semantic_model=semantic_model,
                use_cross_encoder=False
            ), 'recommend'
        ),
        'two_stage_ce_collaborative': (
            TwoStageRecommender(
                candidate_models=[(collab_model, 'recommend')],
                collab_model=collab_model,
                semantic_model=semantic_model,
                use_cross_encoder=True
            ), 'recommend'
        ),
        'two_stage_ce_default': (
            TwoStageRecommender(
                candidate_models=[(collab_model, 'recommend'), (semantic_model, 'recommend_by_hybrid_score')],
                collab_model=collab_model,
                semantic_model=semantic_model,
                use_cross_encoder=True
            ), 'recommend'
        ),
        'two_stage_ce_all_models': (
            TwoStageRecommender(
                candidate_models=[
                    (popularity_model, 'get_top_recommendations'),
                    (collab_model, 'recommend'),
                    (semantic_model, 'recommend_by_hybrid_score')],
                collab_model=collab_model,
                semantic_model=semantic_model,
                use_cross_encoder=True
            ), 'recommend'
        ),
    }

    # === RUN ALL EVALUATIONS ===

    print("\n" + "#" * 50)
    print("RUNNING ALL EVALUATIONS")
    print("#" * 50)

    all_results = {}

    for model_name, (model, method_name) in models_to_test.items():
        try:
            avg_results, coverage = run_evaluation(
                model=model,
                recommend_method_name=method_name,
                ground_truth=ground_truth_data,
                seen_movies=seen_movies_data,
                catalog_ids=catalog_ids
            )

            all_results[model_name] = {
                'precision': avg_results['precision'],
                'recall': avg_results['recall'],
                'ndcg': avg_results['ndcg'],
                'coverage': coverage
            }

        except Exception as e:
            print(f"ERROR evaluating {model_name}: {e}")
            all_results[model_name] = {'error': str(e)}

    # Print summary comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 80)

    results_df = pd.DataFrame(all_results).T
    results_df = results_df.dropna()  # Remove failed evaluations

    # Sort by a key metric (e.g., nDCG)
    if 'ndcg' in results_df.columns:
        results_df = results_df.sort_values('ndcg', ascending=False)

    print(results_df.round(4))

    # Highlight best performing models
    print("\n" + "=" * 50)
    print("TOP PERFORMERS")
    print("=" * 50)

    for metric in ['precision', 'recall', 'ndcg']:
        if metric in results_df.columns:
            valid_results = results_df[results_df[metric] > 0]

            if len(valid_results) > 0:
                best_model = valid_results[metric].idxmax()
                best_score = valid_results.loc[best_model, metric]
                print(f"Best {metric.upper()}: {best_model} ({best_score:.4f})")
            else:
                print(f"No valid results found for {metric.upper()}")
