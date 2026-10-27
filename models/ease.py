import numpy as np
import pandas as pd
from scipy import linalg, sparse


class EASE:
    """Линейная реконструкция положительных взаимодействий, семинар 5."""

    def __init__(self, regularization=100.0):
        if regularization <= 0:
            raise ValueError("Регуляризация EASE должна быть положительной")
        self.regularization = regularization

    def fit(self, train):
        train = train.drop_duplicates(["user_id", "item_id"], keep="last")
        self.users_ = pd.Index(sorted(train.user_id.unique()))
        self.items_ = pd.Index(sorted(train.item_id.unique()))
        positive = train[train.rating >= 4]
        self.history_ = sparse.csr_matrix(
            (np.ones(len(positive), dtype=np.float32), (
                self.users_.get_indexer(positive.user_id),
                self.items_.get_indexer(positive.item_id),
            )), shape=(len(self.users_), len(self.items_)),
        )
        size = len(self.items_)
        gram = (self.history_.T @ self.history_).toarray(order="F")
        gram.flat[::size + 1] += self.regularization
        factor = linalg.cho_factor(gram, lower=True, overwrite_a=True, check_finite=False)
        weights = linalg.cho_solve(
            factor, np.eye(size, dtype=np.float32, order="F"),
            overwrite_b=True, check_finite=False,
        )
        del gram, factor
        # Деление по столбцам; плотные матрицы переиспользуются в памяти.
        weights /= -weights.diagonal().copy()[None, :]
        np.fill_diagonal(weights, 0)
        self.weights_ = np.ascontiguousarray(weights)
        return self

    def predict(self, user_ids, item_ids=None, date=None):
        """Скоры всех товаров либо переданных пар; дата не используется."""
        users = self.users_.get_indexer(user_ids)
        if (users < 0).any():
            raise ValueError("Неизвестный пользователь: используйте общий cold-start")
        if item_ids is None:
            return self.history_[users] @ self.weights_
        items = self.items_.get_indexer(item_ids)
        if len(users) != len(items) or (items < 0).any():
            raise ValueError("Неизвестный товар или несовпадение длин ID")
        scores = np.empty(len(users), dtype=np.float32)
        for row, (user, item) in enumerate(zip(users, items)):
            start, end = self.history_.indptr[user:user + 2]
            scores[row] = (self.history_.data[start:end]
                           @ self.weights_[self.history_.indices[start:end], item])
        return scores
