#!/usr/bin/env python3
"""Populate demo data for movie recommendation system"""
import requests
import random
import time
import sys

API_URL = "http://localhost:8000"
EVENTS_URL = "http://localhost:8001"

# Movie and user ranges
TRENDING_MOVIES = [1, 2, 318, 296, 356, 593, 260, 480, 527, 110,
                   589, 2571, 4993, 5952, 2959, 858, 50, 47, 780, 150]
ALL_MOVIES = list(range(1, 200))
USERS = list(range(1, 100))


def check_services():
    """Verify services are running"""
    try:
        requests.get(f"{API_URL}/health", timeout=5).raise_for_status()
        requests.get(f"{EVENTS_URL}/health", timeout=5).raise_for_status()
        return True
    except:
        return False


def populate():
    """Generate ratings data"""
    # Wait for services
    for _ in range(10):
        if check_services():
            break
        time.sleep(3)
    else:
        print("Services not available")
        sys.exit(1)

    # Generate trending movies (500 ratings for popular movies)
    for _ in range(500):
        requests.post(f"{EVENTS_URL}/interact", json={
            "user_id": random.choice(USERS),
            "movie_id": random.choice(TRENDING_MOVIES),
            "rating": random.choice([4.0, 4.5, 5.0])
        })

    # Generate diverse ratings (1000 ratings across all movies)
    for _ in range(1000):
        requests.post(f"{EVENTS_URL}/interact", json={
            "user_id": random.choice(USERS),
            "movie_id": random.choice(ALL_MOVIES),
            "rating": random.choice([2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        })

    print("Data population complete")


if __name__ == "__main__":
    populate()