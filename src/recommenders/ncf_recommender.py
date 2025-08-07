import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import List, Tuple
import pickle
from src.config import config
from src.data.data_loader import DataLoader


class ImprovedNCFModel(nn.Module):
    """
    NCF that mimics ALS but with neural networks.
    Uses actual ratings (not binary) and similar preprocessing to your ALS.
    """

    def __init__(self, num_users, num_items, embedding_dim=50, hidden_dims=[128, 64]):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Use same embedding size as your ALS (factors=50)
        self.user_embedding_mf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mf = nn.Embedding(num_items, embedding_dim)

        # MLP path (for learning non-linear patterns)
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)

        # MLP layers
        mlp_input_dim = embedding_dim * 2
        mlp_layers = []

        prev_dim = mlp_input_dim
        for hidden_dim in hidden_dims:
            mlp_layers.append(nn.Linear(prev_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(0.1))  # Lower dropout
            prev_dim = hidden_dim

        self.mlp = nn.Sequential(*mlp_layers)

        # Final prediction layer - predicts rating (1-5 scale)
        final_input_dim = embedding_dim + hidden_dims[-1]
        self.final_layer = nn.Sequential(
            nn.Linear(final_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Initialize like matrix factorization
        self._init_weights()

    def _init_weights(self):
        """Initialize similar to matrix factorization"""
        # Small random initialization (like your ALS)
        nn.init.normal_(self.user_embedding_mf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        # Matrix Factorization path (like your ALS)
        user_mf = self.user_embedding_mf(user_ids)
        item_mf = self.item_embedding_mf(item_ids)
        mf_output = user_mf * item_mf  # Element-wise product

        # MLP path (captures non-linear patterns)
        user_mlp = self.user_embedding_mlp(user_ids)
        item_mlp = self.item_embedding_mlp(item_ids)
        mlp_input = torch.cat([user_mlp, item_mlp], dim=-1)
        mlp_output = self.mlp(mlp_input)

        # Combine both paths
        combined = torch.cat([mf_output, mlp_output], dim=-1)
        prediction = self.final_layer(combined)

        return prediction.squeeze()


class ImprovedNCFRecommender:
    """
    NCF that uses the same data preprocessing as your ALS model
    """

    def __init__(self, embedding_dim=50, hidden_dims=[128, 64]):
        # Use same embedding size as your ALS
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.model = None
        self.user_map = None
        self.movie_map = None
        self.inverse_movie_map = None
        self.user_item_matrix = None  # Like your ALS

    def fit(self, epochs=100, batch_size=2048, learning_rate=0.001):
        """Train using the SAME preprocessing as your ALS model"""
        print(f"Training Improved NCF model (ALS-style)...")

        # Use EXACTLY the same preprocessing as your ALS
        loader = DataLoader()
        ratings_df = loader.get_ratings()

        # Same mappings as your ALS
        self.user_map = {id: i for i, id in enumerate(ratings_df['userId'].unique())}
        self.movie_map = {id: i for i, id in enumerate(ratings_df['movieId'].unique())}
        self.inverse_movie_map = {i: id for id, i in self.movie_map.items()}

        num_users = len(self.user_map)
        num_items = len(self.movie_map)

        print(f"Users: {num_users}, Items: {num_items}")
        print(f"Total ratings: {len(ratings_df)} (using ALL ratings like ALS)")

        # Create training data using ALL ratings (like your ALS)
        train_data = self._prepare_rating_data(ratings_df)

        # Create model with same embedding size as your ALS
        self.model = ImprovedNCFModel(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims
        )

        # Train to predict actual ratings
        self._train_model(train_data, epochs, batch_size, learning_rate)

        print("Improved NCF training completed!")

    def _prepare_rating_data(self, ratings_df):
        """Use ALL ratings like your ALS (not just binary likes)"""

        # Map IDs to indices (same as ALS)
        ratings_df = ratings_df.copy()
        ratings_df['user_idx'] = ratings_df['userId'].map(self.user_map)
        ratings_df['movie_idx'] = ratings_df['movieId'].map(self.movie_map)

        # Use ALL ratings with their actual values
        users = ratings_df['user_idx'].values
        items = ratings_df['movie_idx'].values
        ratings = ratings_df['rating'].values

        print(f"Training on {len(ratings)} ratings from {ratings.min():.1f} to {ratings.max():.1f}")

        return {
            'users': torch.LongTensor(users),
            'items': torch.LongTensor(items),
            'ratings': torch.FloatTensor(ratings)
        }

    def _train_model(self, train_data, epochs, batch_size, learning_rate):
        """Train to predict actual ratings (not binary classification)"""

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()  # Regression loss for rating prediction

        num_samples = len(train_data['users'])
        num_batches = (num_samples + batch_size - 1) // batch_size

        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0

            # Shuffle data each epoch
            indices = torch.randperm(num_samples)
            users = train_data['users'][indices]
            items = train_data['items'][indices]
            ratings = train_data['ratings'][indices]

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, num_samples)

                batch_users = users[start_idx:end_idx]
                batch_items = items[start_idx:end_idx]
                batch_ratings = ratings[start_idx:end_idx]

                optimizer.zero_grad()

                predictions = self.model(batch_users, batch_items)
                loss = criterion(predictions, batch_ratings)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if epoch % 20 == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch:3d}, MSE Loss: {avg_loss:.4f}")

    def predict_score(self, user_id: int, movie_id: int) -> float:
        """Predict rating score (like your ALS predict_score)"""
        if user_id not in self.user_map or movie_id not in self.movie_map:
            return 0.0

        user_idx = self.user_map[user_id]
        item_idx = self.movie_map[movie_id]

        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx])
            item_tensor = torch.LongTensor([item_idx])
            score = self.model(user_tensor, item_tensor).item()

        return float(score)

    def recommend(self, user_id: int, n: int = 10, exclude_seen: set = None) -> List[Tuple[int, float]]:
        """Generate recommendations (same interface as your ALS)"""
        if user_id not in self.user_map:
            return []

        exclude_seen = exclude_seen or set()
        user_idx = self.user_map[user_id]

        # Get scores for all items (like your ALS)
        self.model.eval()
        recommendations = []

        with torch.no_grad():
            # Process in batches for efficiency
            batch_size = 1000
            all_item_indices = list(range(len(self.movie_map)))

            for i in range(0, len(all_item_indices), batch_size):
                batch_items = all_item_indices[i:i + batch_size]
                batch_users = [user_idx] * len(batch_items)

                user_tensor = torch.LongTensor(batch_users)
                item_tensor = torch.LongTensor(batch_items)

                scores = self.model(user_tensor, item_tensor).numpy()

                for item_idx, score in zip(batch_items, scores):
                    item_id = self.inverse_movie_map[item_idx]
                    if item_id not in exclude_seen:
                        recommendations.append((item_id, float(score)))

        # Sort by score and return top N (like your ALS)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n]

    def save(self):
        """Save the model"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'user_map': self.user_map,
            'movie_map': self.movie_map,
            'inverse_movie_map': self.inverse_movie_map,
            'embedding_dim': self.embedding_dim,
            'hidden_dims': self.hidden_dims
        }

        torch.save(save_dict, config.paths.CANDIDATE_MODEL_DIR / "improved_ncf_recommender.pt")
        print("Improved NCF model saved successfully!")

    @classmethod
    def load(cls):
        """Load the model"""
        save_dict = torch.load(config.paths.CANDIDATE_MODEL_DIR / "improved_ncf_recommender.pt",
                               map_location='cpu', weights_only=False)

        instance = cls(
            embedding_dim=save_dict['embedding_dim'],
            hidden_dims=save_dict['hidden_dims']
        )

        # Recreate model
        num_users = len(save_dict['user_map'])
        num_items = len(save_dict['movie_map'])

        instance.model = ImprovedNCFModel(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=save_dict['embedding_dim'],
            hidden_dims=save_dict['hidden_dims']
        )

        instance.model.load_state_dict(save_dict['model_state_dict'])
        instance.user_map = save_dict['user_map']
        instance.movie_map = save_dict['movie_map']
        instance.inverse_movie_map = save_dict['inverse_movie_map']

        print("Improved NCF model loaded successfully!")
        return instance

    def cleanup(self):
        """Clean up memory"""
        if self.model is not None:
            del self.model
        torch.cuda.empty_cache()


# Training function
def train_improved_ncf():
    """Train improved NCF that mimics ALS preprocessing"""
    ncf = ImprovedNCFRecommender(
        embedding_dim=50,  # Same as your ALS factors=50
        hidden_dims=[128, 64]
    )
    ncf.fit(epochs=100, batch_size=2048, learning_rate=0.001)
    ncf.save()
    return ncf


if __name__ == "__main__":
    # Train the improved model
    model = train_improved_ncf()

    # Test recommendations
    recommendations = model.recommend(user_id=1, n=10)
    print(f"Recommendations for user 1: {recommendations[:5]}")