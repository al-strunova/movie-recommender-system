# Movie Recommendation System

Production-ready recommendation system with real-time personalization, microservices architecture, and continuous
learning capabilities.

## Key Features

- **Two-Stage Recommendation Pipeline**: Candidate generation + LightGBM/CrossEncoder reranking achieving 56% nDCG
- **Multi-Section Recommendations**: Personalized, Trending, Hidden Gems (GNN), and Genre-Based Cold Start
- **Microservices Architecture**: 6 Docker containers with Redis, RabbitMQ, and FastAPI
- **Smart Cold Start**: Genre-based recommendations with IMDB-weighted ratings achieving 37.5% nDCG
- **Real-time Feedback Processing**: Event-driven architecture with automated retraining
- **Comprehensive Evaluation**: Tested 23 model configurations with multiple metrics

## Architecture

**System Microservices:**

- **Recommendation API** (port 8000): Serves personalized recommendations with Redis caching
- **Event Service** (port 8001): Collects user ratings and publishes to RabbitMQ
- **Event Consumer**: Processes rating events from RabbitMQ and stores in Redis
- **Trending Service**: Calculates popular movies every 30 seconds from Redis data
- **Web UI**: Interactive demo interface served from main API
- **Infrastructure**: Redis (caching), RabbitMQ (message queue)

**Data Flow:**

1. User interacts with Web UI → sends rating to Event Service
2. Event Service → publishes to RabbitMQ queue
3. Event Consumer → processes queue → updates Redis
4. Trending Service → analyzes Redis data → updates trending
5. Recommendation API → reads from Redis → serves recommendations

## Quick Start

### Prerequisites

To run the application, you'll need the pre-trained models and data features.
- Download the models/ and data/ directories from : [Google Drive link](https://drive.google.com/file/d/1b7svWPa_evzUUMybh9GH0cYKKGxVxopk/view?usp=drive_link)
- Extract the contents to the root of the project.

### Using Docker Compose (Recommended)

This is the fastest way to get the entire system running.

```bash
# Clone the repository
git clone https://github.com/yourusername/movie-recsys.git
cd movie-recsys

# Download and extract the pre-trained models into the project root as instructed in the Prerequisites section.

# Build and start all services
docker-compose up --build -d

# Verify all 6 services are running
docker-compose ps

# Populate test data (optional, for trending section)
python scripts/populate_data.py

# Access UI
open http://localhost:8000

# Stop services
docker-compose down
```

### Manual Execution (Development)

This is useful for local development and debugging individual services.

```bash
# Start infrastructure
docker run -d -p 6379:6379 redis:7-alpine
docker run -d -p 5672:5672 rabbitmq:3-alpine

# Start services (in separate terminals)
python services/events/event_consumer.py
python services/pipelines/trending_service.py
uvicorn services.events.event_service:app --port 8001
uvicorn services.recommendations.main:app --port 8000

# Populate test data
python scripts/populate_data.py

# Access UI
open http://localhost:8000
```

## Performance Results

|        Model Configuration                 | Precision@10 | nDCG@10 | Coverage |    Use Case  |
|--------------------------------------------|--------------|---------|----------|--------------|
| **Two-Stage LightGBM + Collaborative**     | 27.1%        | 56.1%   | 7.3%     | Personalized |
| **Two-Stage CrossEncoder + Collaborative** | 22.0%        | 50.7%   | 9.2%     | Personalized |
| Two-Stage LightGBM + GNN                   | 21.8%        | 47.1%   | 2.1%     | Hidden Gems  |
| Two-Stage LightGBM + Genre Popularity      | 14.6%        | 37.5%   | 2.2%     | New Users    |
| Collaborative Filtering (ALS)              | 23.3%        | 52.9%   | 7.5%     | Baseline     |

*Evaluated on MovieLens 100k dataset with temporal split*

## Two-Stage Architecture Deep Dive

**Stage 1 - Candidate Generation:**

This stage generates a list of potential movies for a user from different sources.

- **Collaborative Filtering (ALS)**- Used for users with existing watch history.
- **Graph Neural Network (3-layer GCN)** - Identifies "hidden gems" or related items through an item-item graph.
- **Genre-Based with Weighted Ratings** - A smart approach for new users with no history.
- **Content-based (TF-IDF + SVD)** - An alternative method (tested only).
- **Semantic (Sentence-BERT)** - An alternative method (tested only).

**Stage 2 - Reranking:**

This stage takes the candidate list and re-orders it for optimal results.

- **LightGBM**: A gradient-boosting model with over 25 engineered features. This is the production choice.
- **CrossEncoder transformer**: A transformer model used as an alternative reranker.

**Special Handling:**

- **New users**: Genre preferences + IMDB weighted rating formula
- Weighted rating: (v/(v+m))×R + (m/(v+m))×C where m=vote threshold, C=mean rating

## Technical Stack

- **Machine Learning**: PyTorch, PyTorch Geometric, LightGBM, Sentence-Transformers
- **Infrastructure**: Docker, Docker Compose, FastAPI, Redis, RabbitMQ
- **Data Processing**: Pandas, NumPy, Faiss, scikit-learn
- **Models**: ALS (implicit), GCN, CrossEncoder, TF-IDF/SVD

## Project Structure

```
MovieRecSys/
├── models/                                 # Pre-trained models
│   ├── candidate_models/
│   │   ├── als_model.npz
│   │   ├── als_maps.pkl
│   │   ├── als_user_item_matrix.npz
│   │   ├── gnn_recommender.pt
│   │   ├── genre_model.pkl
│   │   ├── semanticrecommender.pkl
│   │   └── semanticrecommender.faiss
│   └── ranking_models/
│       └── lightgbm_ranker.pkl
├── data/                                    # Datasets and feature files
│   ├── movielens/
│   │   └── tags.csv (required at runtime)
│   ├── imdb/ (training only)
│   └── features/
│       ├── user_features.pkl
│       └── movie_features.pkl
├── src/                                     # Core source code
│   ├── recommenders/                        # Model implementations
│   │   ├── two_stage_recommender.py
│   │   ├── collaborative_model.py
│   │   ├── gnn_recommender.py
│   │   ├── semantic_recommender.py
│   │   ├── content_based_recommender.py
│   │   ├── lightgbm_ranker.py
│   │   ├── cross_encoder_ranker.py
│   │   └── faiss_recommender_abc.py
│   ├── features/                            # Feature engineering
│   └── data/                                # Data loading utilities
├── services/                                # Microservice implementations
│   ├── recommendations/                       # FastAPI service
│   ├── events/                                # Event processing
│   └── pipelines/                             # Retraining & trending
├── evaluation/                              # Model evaluation scripts
├── training/                                # Training scripts
└── docker-compose.yml                       # Defines the microservices architecture
```

## API Endpoints

```
GET  /recommendations/{user_id}           # Personalized recommendations
GET  /recommendations/{user_id}/sections  # Multiple recommendation sections
POST /interact                            # Record user rating
GET  /health                              # Service health check
```

## Dataset

MovieLens 100k enriched with IMDb metadata:

- 943 users, 1,682 movies, 100,000 ratings
- Additional features: genres, tags, directors, actors

## License

MIT License
