from src.recommenders.gnn_recommender import GNNRecommender
from src.config import config


def train_gnn():
    print("Training GNN model...")

    # Create and train GNN
    gnn_model = GNNRecommender()
    gnn_model.fit()

    # Save model
    gnn_model.save()

    print("GNN training completed successfully!")
    return gnn_model


if __name__ == "__main__":
    gnn_model = train_gnn()
