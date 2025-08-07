import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Set
from contextlib import asynccontextmanager
import redis
import json

from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse


class MovieRecommendation(BaseModel):
    movie_id: int
    score: float


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[MovieRecommendation]


class RecommendationSection(BaseModel):
    title: str
    movies: List[MovieRecommendation]


class SectionsResponse(BaseModel):
    user_id: int
    sections: dict[str, RecommendationSection]


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting recommendation API...")

    try:
        from src.recommenders.collaborative_model import CollaborativeFilteringModel
        from src.recommenders.gnn_recommender import GNNRecommender
        from src.recommenders.semantic_recommender import SemanticRecommender
        collab_model = CollaborativeFilteringModel.load()
        semantic_model = SemanticRecommender.load()
        gnn_model = GNNRecommender.load()

        # Main recommender - best performing
        from src.recommenders.two_stage_recommender import TwoStageRecommender
        app.state.main_recommender = TwoStageRecommender(
            candidate_models=[(collab_model, 'recommend')],
            collab_model=collab_model,
            semantic_model=semantic_model,
            use_cross_encoder=False
        )

        # Hidden gems recommender
        app.state.hidden_gems_recommender = TwoStageRecommender(
            candidate_models=[(gnn_model, 'recommend')],
            collab_model=collab_model,
            semantic_model=semantic_model,
            use_cross_encoder=False
        )

        # Cold start
        from src.recommenders.popularity_model import PopularityModel
        popularity_model = PopularityModel.load()
        app.state.cold_start = TwoStageRecommender(
            candidate_models=[(popularity_model, 'get_top_recommendations')],
            collab_model=collab_model,
            semantic_model=semantic_model,
            use_cross_encoder=False
        )

        # Setup Redis
        app.state.redis = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=os.getenv('REDIS_PORT', 6379),
            decode_responses=True
        )

        print("Models and cache loaded successfully")

    except Exception as e:
        raise RuntimeError(f"Startup failed: {e}")

    yield

    print("Shutting down...")
    if hasattr(app.state, 'redis'):
        app.state.redis.close()
    if hasattr(app.state, 'main_recommender'):
        del app.state.main_recommender
    if hasattr(app.state, 'hidden_gems_recommender'):
        del app.state.hidden_gems_recommender
    if hasattr(app.state, 'cold_start'):
        del app.state.cold_start


app = FastAPI(lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo only
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_user_rated_movies(redis_conn, user_id: int) -> Set[int]:
    """Get set of movie IDs the user has already rated"""
    try:
        rated_movies = redis_conn.smembers(f"user:{user_id}:rated")
        return {int(movie_id) for movie_id in rated_movies}
    except Exception as e:
        return set()


def get_cached_recommendations(redis_conn, user_id: int, n: int, exclude_seen: Set[int]):
    """Get recommendations from Redis cache, filtering out seen movies"""
    try:
        cache_key = f"user:{user_id}:recs"
        cached_data = redis_conn.get(cache_key)

        if cached_data:
            recommendations = json.loads(cached_data)
            filtered_recs = [
                (rec["movie_id"], rec["score"])
                for rec in recommendations
                if rec["movie_id"] not in exclude_seen
            ]
            return filtered_recs[:n]
        return None
    except Exception as e:
        return None


def cache_recommendations(redis_conn, user_id: int, recommendations):
    try:
        recs_data = [
            {"movie_id": int(movie_id), "score": float(score)}
            for movie_id, score in recommendations
        ]
        cache_key = f"user:{user_id}:recs"
        redis_conn.set(cache_key, json.dumps(recs_data))
    except:
        pass


def create_response(user_id: int, recommendations):
    return RecommendationResponse(
        user_id=user_id,
        recommendations=[
            MovieRecommendation(movie_id=int(movie_id), score=float(score))
            for movie_id, score in recommendations
        ]
    )


def get_trending_movies(redis_conn) -> List[tuple]:
    """Get trending movies from Redis"""
    try:
        trending = redis_conn.get("trending:movies")
        if trending:
            movie_ids = json.loads(trending)
            # Create descending scores for display
            return [(int(mid), 1.0 - i * 0.01) for i, mid in enumerate(movie_ids[:10])]
        return []
    except Exception as e:
        return []


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("templates/index.html", "r") as f:
        return f.read()


@app.get("/health")
async def health_check():
    """Service health check"""
    checks = {
        "status": "healthy",
        "models_loaded": all([
            hasattr(app.state, 'main_recommender'),
            hasattr(app.state, 'hidden_gems_recommender'),
            hasattr(app.state, 'cold_start')
        ])
    }

    try:
        app.state.redis.ping()
        checks["redis"] = "connected"
    except:
        checks["redis"] = "disconnected"
        checks["status"] = "degraded"

    return checks


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
async def get_recommendations(user_id: int, n: int = 10):
    if not hasattr(app.state, 'main_recommender'):
        raise HTTPException(status_code=503, detail="Service unavailable")

    try:
        # Get user's rated movies for filtering
        exclude_seen = get_user_rated_movies(app.state.redis, user_id)

        # Try to get cached recommendations
        cached_recs = get_cached_recommendations(app.state.redis, user_id, n, exclude_seen)
        if cached_recs:
            return create_response(user_id, cached_recs)

        # Generate new recommendations
        recommendations = app.state.main_recommender.recommend(user_id=user_id, n=n * 2, exclude_seen=exclude_seen)

        # If no recommendations (new user), use cold start
        if not recommendations:
            recommendations = app.state.cold_start.recommend(user_id, n=n * 2, exclude_seen=exclude_seen)

        # Cache the recommendations
        if recommendations:
            cache_recommendations(app.state.redis, user_id, recommendations)

        # Return top n recommendations
        return create_response(user_id, recommendations[:n])

    except Exception as e:
        print(f"Failed to get recommendations for {user_id} user. Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommendations/{user_id}/sections", response_model=SectionsResponse)
async def get_recommendation_sections(user_id: int):
    """Get multiple recommendation sections"""

    try:
        exclude_seen = get_user_rated_movies(app.state.redis, user_id)
        sections = {}

        # 1. For You - Main personalized recommendations
        for_you = app.state.main_recommender.recommend(user_id=user_id, n=10, exclude_seen=exclude_seen)
        if for_you:
            sections["for_you"] = RecommendationSection(
                title="Selected for You",
                movies=[MovieRecommendation(movie_id=m, score=s) for m, s in for_you]
            )

        # 2. Trending Now
        trending = get_trending_movies(app.state.redis)
        if trending:
            sections["trending"] = RecommendationSection(
                title="Trending Now",
                movies=[MovieRecommendation(movie_id=m, score=s) for m, s in trending]
            )

        # 4. Hidden Gems
        hidden_gems = app.state.hidden_gems_recommender.recommend(user_id=user_id, n=10, exclude_seen=exclude_seen)
        if hidden_gems:
            sections["hidden_gems"] = RecommendationSection(
                title="Hidden Gems",
                movies=[MovieRecommendation(movie_id=m, score=s) for m, s in hidden_gems]
            )

        # If user is new, add a popular section
        if not sections:
            popular = app.state.cold_start.recommend(user_id, n=10)
            sections["popular"] = RecommendationSection(
                title="Popular Movies",
                movies=[MovieRecommendation(movie_id=m, score=s) for m, s in popular]
            )

        return SectionsResponse(user_id=user_id, sections=sections)

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/admin/retrain")
async def trigger_retrain():
    """Manual trigger for model retraining"""
    # Just a placeholder - shows you can retrain when needed
    return {"message": "Retraining triggered", "status": "started"}
