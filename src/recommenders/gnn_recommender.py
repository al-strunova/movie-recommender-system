import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from typing import List, Tuple
from src.config import config
from src.data.data_loader import DataLoader


class GNNModel(nn.Module):
    """
    Simple 2-layer GNN model
    """

    def __init__(self, num_features):
        super().__init__()

        # Use config parameters
        hidden_dim = config.params.gnn.hidden_dim
        output_dim = config.params.gnn.output_dim
        dropout = config.params.gnn.dropout

        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        # Layer 2 (NEW)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        # Layer 3 (final)
        x = self.conv3(x, edge_index)
        return x


class GNNRecommender:

    def __init__(self):
        self.model = None
        self.user_map = None
        self.movie_map = None
        self.graph_data = None

    def fit(self):
        """
        Train the GNN model using existing feature engineering
        """
        print("Training GNN model with rich features...")

        # Use existing feature engineering
        from src.features.feature_engineering import FeatureEngineering
        fe = FeatureEngineering()
        user_features_df = fe.build_user_feature_set()
        movie_features_df = fe.build_movie_feature_set()

        print(f"User features: {len(user_features_df)} users")
        print(f"Movie features: {len(movie_features_df)} movies")


        # Create graph data
        self.graph_data = self._create_graph_from_features(user_features_df, movie_features_df)

        # Train model
        self._train_model()

        print("GNN training completed!")

    def _create_graph_from_features(self, user_features_df, movie_features_df):
        """
        Create graph data from existing feature dataframes
        """
        # Get user and movie lists
        users = user_features_df['userId'].values
        movies = movie_features_df['movieId'].values

        # Create node mappings
        self.user_map = {uid: i for i, uid in enumerate(users)}
        self.movie_map = {mid: i + len(users) for i, mid in enumerate(movies)}

        print(f"Created mappings: {len(users)} users, {len(movies)} movies")

        # Convert features to tensors (simple pandas to torch conversion)
        user_features = torch.tensor(
            user_features_df.drop(['userId', 'user_fav_genre'], axis=1, errors='ignore').values,
            dtype=torch.float
        )

        movie_features = torch.tensor(
            movie_features_df.drop(['movieId', 'genres'], axis=1, errors='ignore').values,
            dtype=torch.float
        )

        print(f"User features shape: {user_features.shape}")
        print(f"Movie features shape: {movie_features.shape}")

        # Normalize features
        user_features = (user_features - user_features.mean(dim=0)) / (user_features.std(dim=0) + 1e-8)
        movie_features = (movie_features - movie_features.mean(dim=0)) / (movie_features.std(dim=0) + 1e-8)

        user_dim = user_features.shape[1]
        movie_dim = movie_features.shape[1]

        if user_dim < movie_dim:
            # Pad user features with zeros
            padding = torch.zeros(user_features.shape[0], movie_dim - user_dim)
            user_features = torch.cat([user_features, padding], dim=1)
            print(f"Padded user features to: {user_features.shape}")

        elif movie_dim < user_dim:
            # Pad movie features with zeros
            padding = torch.zeros(movie_features.shape[0], user_dim - movie_dim)
            movie_features = torch.cat([movie_features, padding], dim=1)
            print(f"Padded movie features to: {movie_features.shape}")

        # Combine features
        all_features = torch.cat([user_features, movie_features], dim=0)

        # Create edges from ratings
        edge_index = self._create_edges_from_ratings()

        print(f"Created {edge_index.shape[1]} edges")

        # Create graph
        return Data(
            x=all_features,
            edge_index=edge_index,
            num_nodes=len(users) + len(movies)
        )

    def _create_edges_from_ratings(self):
        """
        Create edges using existing data loader
        """
        loader = DataLoader()
        ratings_df = loader.get_ratings()

        # Filter positive ratings using config threshold
        positive_ratings = ratings_df[ratings_df['rating'] >= config.params.gnn.rating_threshold]

        print(f"Using {len(positive_ratings)} positive ratings as edges")

        edges = []
        for _, row in positive_ratings.iterrows():
            user_id, movie_id = row['userId'], row['movieId']

            # Only include if both user and movie are in our mappings
            if user_id in self.user_map and movie_id in self.movie_map:
                user_node = self.user_map[user_id]
                movie_node = self.movie_map[movie_id]

                # Bidirectional edges
                edges.extend([[user_node, movie_node], [movie_node, user_node]])

        return torch.tensor(edges, dtype=torch.long).t()

    def _train_model(self):
        """
        Train the GNN model using BCE loss
        """
        # Create model
        num_features = self.graph_data.x.shape[1]
        self.model = GNNModel(num_features)

        print(f"Model: {num_features} → {config.params.gnn.hidden_dim} → {config.params.gnn.output_dim}")

        # Use config parameters
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=config.params.gnn.learning_rate,
                                     weight_decay=1e-5)
        num_epochs = config.params.gnn.num_epochs

        print(f"Training for {num_epochs} epochs with BCE loss...")

        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            embeddings = self.model(self.graph_data.x, self.graph_data.edge_index)
            loss = self._compute_bce_loss(embeddings, self.graph_data.edge_index)

            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}")

    def _compute_bce_loss(self, embeddings, edge_index):
        """
        Compute Binary Cross Entropy loss with negative sampling
        """
        # Positive edges (actual connections)
        pos_scores = (embeddings[edge_index[0]] * embeddings[edge_index[1]]).sum(dim=1)

        # Sample negative edges
        num_nodes = embeddings.size(0)
        num_neg = edge_index.size(1)
        neg_edges = torch.randint(0, num_nodes, (2, num_neg))
        neg_scores = (embeddings[neg_edges[0]] * embeddings[neg_edges[1]]).sum(dim=1)

        # Create labels and compute BCE loss
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([
            torch.ones(len(pos_scores)),
            torch.zeros(len(neg_scores))
        ])

        return F.binary_cross_entropy_with_logits(scores, labels)

    def recommend(self, user_id: int, n: int = 10, exclude_seen: set = None) -> List[Tuple[int, float]]:
        """
        Get recommendations for a user
        """
        if user_id not in self.user_map:
            return []

        exclude_seen = exclude_seen or set()

        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(self.graph_data.x, self.graph_data.edge_index)

        user_node = self.user_map[user_id]
        user_embedding = embeddings[user_node]

        # Get movie embeddings
        num_users = len(self.user_map)
        movie_embeddings = embeddings[num_users:]

        # Calculate scores using dot product (learned from BCE loss)
        raw_scores = torch.sum(user_embedding.unsqueeze(0) * movie_embeddings, dim=1)

        temperature = 1.2  # This was working well
        scaled_scores = raw_scores / temperature
        scores = torch.softmax(scaled_scores, dim=0)

        # Get recommendations
        movie_ids = list(self.movie_map.keys())
        recommendations = []

        for i, score in enumerate(scores):
            movie_id = movie_ids[i]
            if movie_id not in exclude_seen:
                recommendations.append((movie_id, score.item()))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n]

    def save(self):
        """
        Save model using config path
        """
        if self.model is None:
            raise ValueError("No model to save!")

        save_dict = {
            'model_state': self.model.state_dict(),
            'user_map': self.user_map,
            'movie_map': self.movie_map,
            'num_features': self.graph_data.x.shape[1],
            'graph_data': self.graph_data
        }
        torch.save(save_dict, config.paths.GNN_MODEL_PATH)
        print(f"GNN model saved to {config.paths.GNN_MODEL_PATH}")

    @classmethod
    def load(cls):
        """
        Load model using config path
        """
        save_dict = torch.load(config.paths.GNN_MODEL_PATH, map_location='cpu', weights_only=False)

        instance = cls()
        instance.user_map = save_dict['user_map']
        instance.movie_map = save_dict['movie_map']
        instance.graph_data = save_dict['graph_data']

        # Recreate model
        instance.model = GNNModel(save_dict['num_features'])
        instance.model.load_state_dict(save_dict['model_state'])

        print(f"GNN model loaded successfully")
        return instance

    def cleanup(self):
        """
        Clean up memory
        """
        if self.model is not None:
            del self.model
        if self.graph_data is not None:
            del self.graph_data
        torch.cuda.empty_cache()