import os
import json
import redis
import time
from collections import Counter
from datetime import datetime, timedelta

redis_conn = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    decode_responses=True
)


def calculate_trending():
    """Count movie ratings from the last 24 hours"""
    movie_counts = Counter()

    # Get all user ratings
    for key in redis_conn.scan_iter("user:*:ratings"):
        ratings = redis_conn.hgetall(key)

        for movie_id, rating_data in ratings.items():
            movie_counts[movie_id] += 1

    # Get top 20 movies
    trending_movies = [movie_id for movie_id, count in movie_counts.most_common(20)]

    # Store in Redis
    if trending_movies:
        redis_conn.set("trending:movies", json.dumps(trending_movies))
        print(f"Updated trending: {len(trending_movies)} movies")
    else:
        print("No trending movies found")


def main():
    """Update trending every 30 seconds"""
    print("Starting trending service...")

    try:
        redis_conn.ping()
        print("Redis connected successfully")
    except Exception as e:
        print(f"Redis connection failed: {e}")
        return

    # Run first calculation immediately
    calculate_trending()

    # Then update every 30 seconds
    while True:
        time.sleep(30)
        try:
            calculate_trending()
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()