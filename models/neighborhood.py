import numpy as np
import pandas as pd
from scipy import sparse


class NeighborhoodCF:
    """User/item CF: прогноз оценки и скор ранжирования как в семинаре 3."""

    def __init__(self, kind="item", similarity="centered_cosine", n_neighbors=40,
                 min_common=3, shrinkage=10, block_size=512, min_neighbors=3, confidence=5):
        if kind not in {"user", "item"}:
            raise ValueError("kind: user или item")
        if similarity not in {"cosine", "centered_cosine", "binary_cosine", "jaccard"}:
            raise ValueError("Неизвестная мера сходства")
        if min(n_neighbors, min_common, block_size, min_neighbors) < 1 or min(shrinkage, confidence) < 0:
            raise ValueError("Некорректные параметры соседей")
        self.kind, self.similarity = kind, similarity
        self.n_neighbors, self.min_common = n_neighbors, min_common
        self.shrinkage, self.block_size = shrinkage, block_size
        self.min_neighbors, self.confidence = min_neighbors, confidence

    def fit(self, train):
        train = train.drop_duplicates(["user_id", "item_id"], keep="last")
        self.users_ = pd.Index(sorted(train["user_id"].unique()))
        self.items_ = pd.Index(sorted(train["item_id"].unique()))
        u = self.users_.get_indexer(train["user_id"])
        i = self.items_.get_indexer(train["item_id"])
        ratings = sparse.csr_matrix(
            (train["rating"].to_numpy(np.float32), (u, i)),
            shape=(len(self.users_), len(self.items_)),
        )
        observed = ratings.copy()
        observed.data[:] = 1
        counts = np.diff(ratings.indptr)
        self.means_ = np.asarray(ratings.sum(axis=1)).ravel() / np.maximum(counts, 1)
        self.centered_ = ratings.copy()
        self.centered_.data -= np.repeat(self.means_, counts)
        positive = (ratings >= 4).astype(np.float32)
        features = ratings
        if self.similarity == "centered_cosine":
            features = self.centered_
        elif self.similarity in {"binary_cosine", "jaccard"}:
            features, observed = positive, positive
        if self.kind == "item":
            features, observed = features.T, observed.T
        features, observed = features.tocsr(), observed.tocsr()
        norms = np.sqrt(np.asarray(features.multiply(features).sum(axis=1)).ravel())
        degrees = np.asarray(observed.sum(axis=1)).ravel()
        blocks = []
        for start in range(0, features.shape[0], self.block_size):
            end = min(start + self.block_size, features.shape[0])
            common = (observed[start:end] @ observed.T).tocsr()
            common.data[common.data < self.min_common] = 0
            common.eliminate_zeros()
            products = common if self.similarity == "jaccard" else features[start:end] @ features.T
            pairs = products.multiply(common.astype(bool)).tocoo()
            overlap = np.asarray(common[pairs.row, pairs.col]).ravel()
            denominator = (degrees[start + pairs.row] + degrees[pairs.col] - overlap
                           if self.similarity == "jaccard"
                           else norms[start + pairs.row] * norms[pairs.col])
            similarity = np.divide(pairs.data, denominator, out=np.zeros_like(pairs.data),
                                   where=denominator > 0)
            similarity *= overlap / (overlap + self.shrinkage)
            keep = (similarity > 0) & (start + pairs.row != pairs.col)
            blocks.append(sparse.csr_matrix(
                (similarity[keep], (pairs.row[keep], pairs.col[keep])), shape=common.shape,
            ))
        self.similarity_ = sparse.vstack(blocks, format="csr")
        self.similarity_csc_ = self.similarity_.tocsc() if self.kind == "item" else None
        return self

    def set_neighbors(self, k):
        if k < 1:
            raise ValueError("k должен быть положительным")
        self.n_neighbors = k
        return self

    def _predict_one(self, user, items=None):
        size = len(self.items_) if items is None else len(items)
        if self.kind == "item":
            history = self.centered_.getrow(user)
            weights = self.similarity_csc_[:, history.indices]
            if items is not None:
                weights = weights[items]
            if len(history.indices) > self.n_neighbors:
                weights = weights.toarray()
                order = np.argpartition(weights, -self.n_neighbors, axis=1)[:, -self.n_neighbors:]
                weights = np.take_along_axis(weights, order, axis=1)
                numerator = (weights * history.data[order]).sum(axis=1)
                denominator = weights.sum(axis=1)
                support = (weights > 0).sum(axis=1)
            else:
                numerator = np.asarray(weights @ history.data).ravel()
                denominator = np.asarray(weights.sum(axis=1)).ravel()
                support = np.asarray(weights.getnnz(axis=1)).ravel()
        else:
            neighbors = self.similarity_.getrow(user)
            order = np.argsort(-neighbors.data, kind="stable")
            weights, neighbor_ids = neighbors.data[order], neighbors.indices[order]
            residuals = self.centered_[neighbor_ids].tocsc()
            if items is not None:
                residuals = residuals[:, items]
            residuals.sort_indices()
            counts = np.diff(residuals.indptr)
            columns = np.repeat(np.arange(size), counts)
            positions = np.arange(residuals.nnz) - np.repeat(residuals.indptr[:-1], counts)
            keep = positions < self.n_neighbors
            selected_weights = weights[residuals.indices[keep]]
            numerator = np.bincount(columns[keep], weights=selected_weights * residuals.data[keep], minlength=size)
            denominator = np.bincount(columns[keep], weights=selected_weights, minlength=size)
            support = np.bincount(columns[keep], minlength=size)
        correction = np.divide(numerator, denominator, out=np.zeros(size), where=denominator > 0)
        return np.clip(self.means_[user] + correction, 1, 5).astype(np.float32), support.astype(np.int16)

    def predict(self, user_ids, item_ids=None, return_support=False, date=None):
        """Прогнозы оценок: для всех товаров либо переданных пар."""
        users = self.users_.get_indexer(user_ids)
        if (users < 0).any():
            raise ValueError("Неизвестный пользователь: используйте общий cold-start")
        if item_ids is None:
            values = [self._predict_one(user) for user in users]
            prediction = np.vstack([value[0] for value in values])
            support = np.vstack([value[1] for value in values])
        else:
            items = self.items_.get_indexer(item_ids)
            if len(users) != len(items) or (items < 0).any():
                raise ValueError("Неизвестный товар или несовпадение длин ID")
            prediction = np.empty(len(users), dtype=np.float32)
            support = np.empty(len(users), dtype=np.int16)
            for user in np.unique(users):
                mask = users == user
                prediction[mask], support[mask] = self._predict_one(user, items[mask])
        return (prediction, support) if return_support else prediction

    def rank(self, user_ids, item_ids=None, return_support=False, date=None):
        """Поправка на число соседей из recommend в семинаре 3."""
        prediction, support = self.predict(user_ids, item_ids, return_support=True)
        baseline = self.means_[self.users_.get_indexer(user_ids)]
        if item_ids is None:
            baseline = baseline[:, None]
        weight = np.divide(support, support + self.confidence, out=np.zeros_like(prediction),
                           where=support + self.confidence > 0)
        score = baseline + (prediction - baseline) * weight
        score = np.where(support >= self.min_neighbors, score, -np.inf)
        return (score, support) if return_support else score
