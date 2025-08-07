import os
import json
import redis
import pandas as pd
from datetime import datetime
from pathlib import Path

from src.data.data_loader import DataLoader
from src.recommenders.collaborative_model import CollaborativeFilteringModel


class TrainingPipeline:
    def __init__(self):
        self.redis_conn = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            decode_responses=True
        )
        self.loader = DataLoader()

    def collect_redis_ratings(self):
        """Collect all ratings from Redis"""
        print("Collecting ratings from Redis...")
        new_ratings = []

        # Scan all user keys
        for key in self.redis_conn.scan_iter("user:*:ratings"):
            user_id = int(key.split(":")[1])
            ratings = self.redis_conn.hgetall(key)

            for movie_id, rating_info in ratings.items():
                rating_data = json.loads(rating_info)
                new_ratings.append({
                    'userId': user_id,
                    'movieId': int(movie_id),
                    'rating': float(rating_data['rating']),
                    'timestamp': rating_data['timestamp']
                })

        print(f"Collected {len(new_ratings)} ratings from Redis")
        return pd.DataFrame(new_ratings)

    def merge_ratings(self, existing_ratings_df, new_ratings_df):
        """Merge existing ratings with new ratings from Redis"""
        if new_ratings_df.empty:
            return existing_ratings_df

        # Combine datasets
        combined = pd.concat([existing_ratings_df, new_ratings_df], ignore_index=True)

        # Remove duplicates, keeping the most recent rating
        combined = combined.sort_values('timestamp', ascending=False)
        combined = combined.drop_duplicates(subset=['userId', 'movieId'], keep='first')

        return combined

    def train_collaborative_model(self, ratings_df):
        """Train collaborative filtering model"""
        print("Training collaborative filtering model...")

        # Save the merged ratings for the model to load
        temp_ratings_path = Path("data/merged_ratings.csv")
        ratings_df.to_csv(temp_ratings_path, index=False)

        # Train the model
        collab_model = CollaborativeFilteringModel()
        collab_model.fit()
        collab_model.save()

        print("Collaborative model trained and saved")
        return collab_model

    def invalidate_caches(self):
        """Invalidate all recommendation caches"""
        print("Invalidating recommendation caches...")

        keys_deleted = 0
        for key in self.redis_conn.scan_iter("user:*:recs"):
            self.redis_conn.delete(key)
            keys_deleted += 1

        print(f"Deleted {keys_deleted} cache entries")

    def run_training(self):
        """Run the complete training pipeline"""
        print(f"Starting training pipeline at {datetime.now()}")

        try:
            # Load existing ratings
            existing_ratings = self.loader.get_ratings()

            # Collect new ratings from Redis
            new_ratings = self.collect_redis_ratings()

            # Merge ratings
            merged_ratings = self.merge_ratings(existing_ratings, new_ratings)

            # Train models
            self.train_collaborative_model(merged_ratings)

            # Invalidate caches
            self.invalidate_caches()

            # Log training completion
            self.redis_conn.set(
                "training:last_run",
                json.dumps({
                    "timestamp": datetime.now().isoformat(),
                    "ratings_count": len(merged_ratings),
                    "new_ratings_count": len(new_ratings)
                })
            )

            print(f"Training pipeline completed successfully at {datetime.now()}")

        except Exception as e:
            print(f"Training pipeline failed: {e}")
            raise


def run_scheduled_training():
    """Run training on schedule (daily/weekly)"""
    pipeline = TrainingPipeline()
    pipeline.run_training()


if __name__ == "__main__":
    # For manual runs
    run_scheduled_training()