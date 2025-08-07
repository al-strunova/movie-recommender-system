import os

from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timezone
import pika
import json

from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RatingEvent(BaseModel):
    user_id: int
    movie_id: int
    rating: float


def get_rabbitmq_connection():
    connection = pika.BlockingConnection(pika.ConnectionParameters(os.getenv('RABBITMQ_HOST', 'localhost')))
    channel = connection.channel()
    channel.queue_declare(queue='rating_events', durable=True)
    return connection, channel


@app.post("/interact")
async def record_rating(event: RatingEvent):
    # Validate rating
    if not (1.0 <= event.rating <= 5.0):
        return {"error": "rating must be between 1.0 and 5.0"}

    # Create event data
    event_data = {
        "user_id": event.user_id,
        "movie_id": event.movie_id,
        "rating": event.rating,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

    # Send to RabbitMQ
    try:
        connection, channel = get_rabbitmq_connection()
        channel.basic_publish(
            exchange='',
            routing_key='rating_events',
            body=json.dumps(event_data),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        connection.close()

        return {
            "status": "success",
            "message": "Rating recorded",
            "user_id": event.user_id,
            "movie_id": event.movie_id,
            "rating": event.rating
        }

    except Exception as e:
        return {"error": f"Failed to record rating: {str(e)}"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}