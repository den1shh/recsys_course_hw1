import numpy as np
import pandas as pd
from scipy import sparse


class ExplicitALS:
    """ALS по наблюдаемым оценкам с L2-регуляризацией, семинар 4."""

    def __init__(self, n_factors=16, regularization=5.0, n_epochs=5, seed=42):
        if n_factors < 1 or n_epochs < 1 or regularization <= 0:
            raise ValueError("Параметры ALS должны быть положительными")
        self.n_factors = n_factors
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.seed = seed

    def fit(self, train):
        train = train.drop_duplicates(["user_id", "item_id"], keep="last")
        self.users_ = pd.Index(sorted(train.user_id.unique()))
        self.items_ = pd.Index(sorted(train.item_id.unique()))
        users = self.users_.get_indexer(train.user_id)
        items = self.items_.get_indexer(train.item_id)
        ratings = sparse.csr_matrix(
            (train.rating.to_numpy(float), (users, items)),
            shape=(len(self.users_), len(self.items_)),
        )
        rng = np.random.default_rng(self.seed)
        self.user_factors_ = rng.normal(0, 0.1, (len(self.users_), self.n_factors))
        self.item_factors_ = rng.normal(0, 0.1, (len(self.items_), self.n_factors))
        by_item = ratings.T.tocsr()
        penalty = self.regularization * np.eye(self.n_factors)

        for _ in range(self.n_epochs):
            for matrix, factors, other in [
                (ratings, self.user_factors_, self.item_factors_),
                (by_item, self.item_factors_, self.user_factors_),
            ]:
                for row in range(matrix.shape[0]):
                    start, end = matrix.indptr[row:row + 2]
                    values = other[matrix.indices[start:end]]
                    factors[row] = np.linalg.solve(
                        values.T @ values + penalty,
                        values.T @ matrix.data[start:end],
                    )
        return self

    def predict(self, user_ids, item_ids=None, date=None):
        """Скоры всех товаров либо переданных пар; дата не используется."""
        users = self.users_.get_indexer(user_ids)
        if (users < 0).any():
            raise ValueError("Неизвестный пользователь: используйте общий cold-start")
        if item_ids is None:
            return self.user_factors_[users] @ self.item_factors_.T
        items = self.items_.get_indexer(item_ids)
        if len(users) != len(items) or (items < 0).any():
            raise ValueError("Неизвестный товар или несовпадение длин ID")
        return np.sum(self.user_factors_[users] * self.item_factors_[items], axis=1)
