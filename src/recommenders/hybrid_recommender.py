from src.recommenders.popularity_model import PopularityModel
from src.recommenders.collaborative_model import CollaborativeFilteringModel
from src.config import config


class HybridRecommender:
    """
    A hybrid recommender that intelligently switches between a personalized
    collaborative filtering model and a non-personalized popularity model.
    """

    def __init__(self, collaborative_model: CollaborativeFilteringModel, popularity_model: PopularityModel):
        """
        Initializes the HybridRecommender with fitted instances of the models it depends on.
        """
        self.collaborative_model = collaborative_model
        self.popularity_model = popularity_model
        # To get the list of "known" users from the collaborative model's map
        self.known_users = set(self.collaborative_model.user_map.keys())

    def recommend(self, user_id: int,
                  n: int = config.params.recommender.n_final_recommendations,
                  exclude_seen: set = None) -> list:
        """
        Generates recommendations for a user.

        It first checks if the user is known to the collaborative filtering model.
        If so, it uses that model. If not, it falls back to the popularity model.
        """
        # Check if the user is "known" (i.e., was part of the training data)
        if user_id in self.known_users:
            return self.collaborative_model.recommend(user_id, n=n, exclude_seen=exclude_seen)
        else:
            # Use the static top-N recommendations for the fallback
            return self.popularity_model.get_top_recommendations(user_id, n=n, exclude_seen=exclude_seen)