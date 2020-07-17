import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from abc import ABCMeta, abstractmethod

# TODO: Conserve memory by setting numpy data types and benchmark performance


class RecommenderBase(BaseEstimator, metaclass=ABCMeta):
    """
    Abstract base class for all recommender models.
    All subclasses should implement the fit() and predict() methods

    Arguments:
        min_rating {int} -- Smallest rating possible (default: {0})
        max_rating {int} -- Largest rating possible (default: {5})
        verbose {str} -- Verbosity when fitting. Values possible are 0 to not print anything, 1 to print fitting model (default: {1})

    Attributes:
        n_users {int} -- Number of users
        n_items {int} -- Number of items
        global_mean {float} -- Global mean of all ratings
        user_id_map {dict} -- Mapping of user ids to assigned integer ids
        item_id_map {dict} -- Mapping of item ids to assigned integer ids

    """

    def __init__(self, min_rating: float = 0, max_rating: float = 5, verbose: int = 0):
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.verbose = verbose
        self.n_users, self.n_items = None, None
        self.global_mean = None
        self.user_id_map, self.item_id_map = None, None
        return

    def _preprocess_data(self, X: pd.DataFrame, type: str = "fit") -> pd.DataFrame:
        """
        Preprocessing steps before doing fit, update or predict

        Arguments:
            X {pd.DataFrame} -- Dataframe containing columns user_id, item_id and rating
            type {str} -- The type of preprocessing to do. Allowed options are ('fit', 'predict', 'update'). Defaults to 'fit'

        Returns:
            X [pd.DataFrame] -- Dataframe with columns user_id, item_id and rating
            known_users [list, 'on update only'] -- List containing already known users in X. Only returned for type update
            new_users [list, 'on update only'] -- List containing new users in X. Only returned for type update
        """
        # Keep only required columns in given order
        if type == "predict":
            cols_keep = ["user_id", "item_id"]
        else:
            cols_keep = ["user_id", "item_id", "rating"]

        X = X.loc[:, cols_keep]

        if type in ("fit", "update"):
            # Check for duplicate user-item ratings
            if X.duplicated(subset=["user_id", "item_id"]).sum() != 0:
                raise ValueError("Duplicate user-item ratings in matrix")

            # Shuffle rows
            X = X.sample(frac=1, replace=False)

        if type == "fit":
            # Create mapping of user_id and item_id to assigned integer ids
            user_ids = X["user_id"].unique()
            item_ids = X["item_id"].unique()
            self.user_id_map = {user_id: i for (i, user_id) in enumerate(user_ids)}
            self.item_id_map = {item_id: i for (i, item_id) in enumerate(item_ids)}
            self.n_users = len(user_ids)
            self.n_items = len(item_ids)

        elif type == "update":
            # Keep only item ratings for which the item is already known
            items = self.item_id_map.keys()
            X = X.query("item_id in @items").copy()

            # Add information on new users
            new_users, known_users = [], []
            users = X["user_id"].unique()
            new_user_id = max(self.user_id_map.values()) + 1

            for user in users:
                if user in self.user_id_map.keys():
                    known_users.append(user)
                    continue

                # Add to user id mapping
                new_users.append(user)
                self.user_id_map[user] = new_user_id
                new_user_id += 1

        # Remap user id and item id to assigned integer ids
        X.loc[:, "user_id"] = X["user_id"].map(self.user_id_map)
        X.loc[:, "item_id"] = X["item_id"].map(self.item_id_map)

        if type == "predict":
            # Replace missing mappings with -1
            X.fillna(-1, inplace=True)

        if type == "update":
            return X, known_users, new_users
        else:
            return X

    @abstractmethod
    def fit(self, X: pd.DataFrame):
        """
        Fit model to given data

        Args:
            X {pandas DataFrame} -- Dataframe containing columns user_id, item_id and rating
        """
        return self

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> list:
        """
        Predict ratings for given users and items

        Args:
            X (pd.DataFrame): Dataframe containing columns user_id and item_id

        Returns:
            list: List containing rating predictions of all user, items in same order as input X
        """
        return []

    def recommend(
        self,
        user,
        amount: int = 10,
        items_known: list = None,
        include_user: bool = True,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame of recommendations of items for a given user sorted from highest to lowest.

        Args:
            user (any): User_id to get recommendations for (not assigned user_id from self.user_id_map)
            items_known (list, optional): List of items already known by user and to not be considered in recommendations. Defaults to None.
            include_user (bool, optional): Whether to include the user_id in the output DataFrame or not. Defaults to True.

        Returns:
            pd.DataFrame: Recommendations DataFrame for user with columns user_id (optional), item_id, rating sorted from highest to lowest rating 
        """
        items = list(self.item_id_map.keys())

        # If items_known is provided then filter by items that the user does not know
        if items_known is not None:
            items_known = list(items_known)
            items = [item for item in items if item not in items_known]

        # Get rating predictions for given user and all unknown items
        items_recommend = pd.DataFrame({"user_id": user, "item_id": items})
        items_recommend["rating_pred"] = self.predict(X=items_recommend)

        # Sort and keep top n items
        items_recommend.sort_values(by="rating_pred", ascending=False, inplace=True)
        items_recommend = items_recommend.head(amount)

        if not include_user:
            items_recommend.drop(["user_id"], axis="columns", inplace=True)

        return items_recommend

