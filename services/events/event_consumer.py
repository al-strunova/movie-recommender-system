import os
import pika
import json
import redis
from dotenv import load_dotenv

load_dotenv()


def get_redis_connection():
    return redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=os.getenv('REDIS_PORT', 6379),
        decode_responses=True
    )


def get_rabbitmq_connection():
    host = os.getenv('RABBITMQ_HOST', 'localhost')
    connection = pika.BlockingConnection(pika.ConnectionParameters(host))
    channel = connection.channel()
    channel.queue_declare(queue='rating_events', durable=True)
    return connection, channel


def process_rating_event(event_data, redis_conn):
    try:
        user_id = event_data['user_id']
        movie_id = event_data['movie_id']
        rating = event_data['rating']
        timestamp = event_data['timestamp']

        # Store in user's rated movies set (for exclusion)
        redis_conn.sadd(f"user:{user_id}:rated", movie_id)

        # Store in user's ratings hash (for retrieval)
        redis_conn.hset(
            f"user:{user_id}:ratings",
            movie_id,
            json.dumps({"rating": rating, "timestamp": timestamp})
        )

        print(f"Processed rating: user={user_id}, movie={movie_id}, rating={rating}")

    except Exception as e:
        print(f"Error processing rating event: {e}")


def start_consumer():
    redis_conn = get_redis_connection()
    connection, channel = get_rabbitmq_connection()

    def callback(ch, method, properties, body):
        try:
            event_data = json.loads(body)
            process_rating_event(event_data, redis_conn)
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            print(f"Error in callback: {e}")
            ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='rating_events', on_message_callback=callback)

    print("Starting rating consumer...")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("Stopping consumer...")
        channel.stop_consuming()
        connection.close()


if __name__ == "__main__":
    start_consumer()
