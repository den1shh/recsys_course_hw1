from collections import Counter
from html import unescape
import re

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit


class FactorizationMachine:
    """FM второго порядка с метаданными и BPR из семинара 5."""

    def __init__(self, n_factors=8, epochs=2, lr=0.01, reg=1e-5,
                 batch_size=4096, max_features=128, min_df=2,
                 use_metadata=True, random_state=42):
        self.n_factors, self.epochs, self.lr = n_factors, epochs, lr
        self.reg, self.batch_size = reg, batch_size
        self.max_features, self.min_df = max_features, min_df
        self.use_metadata, self.random_state = use_metadata, random_state

    def _build_features(self, items):
        metadata = items.drop_duplicates("item_id").set_index("item_id").reindex(self.items_)
        documents, categories = [], []
        for row in metadata.to_dict("records"):
            text, labels = [], set()
            for column in ("title", "description", "feature", "brand", "category",
                           "platform", "os", "color", "wireless"):
                value = row.get(column, "")
                values = value if isinstance(value, list) else ([] if pd.isna(value) else [value])
                values = [unescape(str(value)).strip().lower() for value in values if value is not None]
                if column in {"title", "description", "feature"}:
                    text.extend(values)
                else:
                    labels.update((column, value) for value in values if value)
            text = re.sub(r"<[^>]*>", " ", " ".join(text))
            documents.append(Counter(re.findall(r"(?u)\b\w\w+\b", text)))
            categories.append(labels)
        df, counts = Counter(), Counter()
        for document in documents:
            df.update(document.keys())
            counts.update(document)
        words = sorted((word for word in df if df[word] >= self.min_df),
                       key=lambda word: (-counts[word], word))[:self.max_features]
        self.vocabulary_ = {word: i for i, word in enumerate(sorted(words))} if self.use_metadata else {}
        self.idf_ = np.array([np.log((len(self.items_) + 1) / (df[word] + 1)) + 1
                              for word in self.vocabulary_], dtype=np.float32)
        labels = sorted(set().union(*categories)) if self.use_metadata else []
        categorical = {label: i + len(self.vocabulary_) for i, label in enumerate(labels)}
        rows, columns, values = [], [], []
        for i, (document, category) in enumerate(zip(documents, categories)):
            terms = [(self.vocabulary_[word], count * self.idf_[self.vocabulary_[word]])
                     for word, count in document.items() if word in self.vocabulary_]
            norm = np.sqrt(sum(value ** 2 for _, value in terms))
            for column, value in terms:
                rows.append(i)
                columns.append(column)
                values.append(value / norm)
            for label in category:
                if label in categorical:
                    rows.append(i)
                    columns.append(categorical[label])
                    values.append(1)
        side = sparse.csr_matrix((values, (rows, columns)), dtype=np.float32,
                                 shape=(len(self.items_), len(self.vocabulary_) + len(labels)))
        norm = np.sqrt(np.asarray(side.multiply(side).sum(axis=1)).ravel())
        side = sparse.diags(np.divide(1, norm, out=np.zeros_like(norm), where=norm > 0)) @ side
        self.user_features_ = sparse.eye(len(self.users_), format="csr", dtype=np.float32)
        self.item_features_ = sparse.hstack([
            sparse.eye(len(self.items_), format="csr", dtype=np.float32), side,
        ], format="csr")
        self.feature_names_ = ([f"user={value}" for value in self.users_]
                               + [f"item={value}" for value in self.items_]
                               + [f"text={word}" for word in self.vocabulary_]
                               + [f"{column}={value}" for column, value in labels])

    def _pairs(self, users, items):
        return sparse.hstack([self.user_features_[users], self.item_features_[items]], format="csr")

    def _forward(self, X):
        squared, projected = X.multiply(X), X @ self.V_
        score = X @ self.w_ + 0.5 * np.sum(projected ** 2 - squared @ (self.V_ ** 2), axis=1)
        return np.asarray(score).ravel(), projected, squared

    def _gradients(self, X, projected, squared, error):
        gw = np.asarray(X.T @ error).ravel()
        gv = X.T @ (error[:, None] * projected) - np.asarray(squared.T @ error).ravel()[:, None] * self.V_
        return gw, np.asarray(gv)

    def _bpr_gradients(self, Xp, Xn):
        positive, projected_p, squared_p = self._forward(Xp)
        negative, projected_n, squared_n = self._forward(Xn)
        difference = positive - negative
        weight = expit(-difference)
        wp, vp = self._gradients(Xp, projected_p, squared_p, -weight)
        wn, vn = self._gradients(Xn, projected_n, squared_n, weight)
        loss = np.logaddexp(0, -difference).mean()
        loss += 0.5 * self.reg * (np.sum(self.w_ ** 2) + np.sum(self.V_ ** 2))
        return float(loss), (wp + wn) / Xp.shape[0] + self.reg * self.w_, \
            (vp + vn) / Xp.shape[0] + self.reg * self.V_

    def _sample_negatives(self, users, rng):
        negative = rng.integers(len(self.items_), size=len(users))
        collision = np.asarray(self.seen_[users, negative]).ravel().astype(bool)
        for _ in range(10):
            if not collision.any():
                return negative
            negative[collision] = rng.integers(len(self.items_), size=collision.sum())
            collision = np.asarray(self.seen_[users, negative]).ravel().astype(bool)
        # Почти полный каталог: выбираем из остатка без долгих повторов.
        for user in np.unique(users[collision]):
            seen = self.seen_.indices[self.seen_.indptr[user]:self.seen_.indptr[user + 1]]
            available = np.setdiff1d(np.arange(len(self.items_)), seen, assume_unique=True)
            mask = collision & (users == user)
            negative[mask] = rng.choice(available, size=mask.sum())
        return negative

    def fit(self, train, items):
        train = train.drop_duplicates(["user_id", "item_id"], keep="last")
        self.users_ = pd.Index(sorted(train["user_id"].unique()))
        self.items_ = pd.Index(sorted(train["item_id"].unique()))
        self._build_features(items)
        users, objects = self.users_.get_indexer(train["user_id"]), self.items_.get_indexer(train["item_id"])
        self.seen_ = sparse.csr_matrix((np.ones(len(train), dtype=bool), (users, objects)),
                                       shape=(len(self.users_), len(self.items_)))
        positive = (train["rating"].to_numpy() >= 4) & (np.diff(self.seen_.indptr)[users] < len(self.items_))
        users, objects = users[positive], objects[positive]
        rng = np.random.default_rng(self.random_state)
        self.w_ = np.zeros(len(self.feature_names_), dtype=np.float32)
        self.V_ = (0.05 * rng.standard_normal((len(self.w_), self.n_factors))).astype(np.float32)
        moments = [(np.zeros_like(value), np.zeros_like(value)) for value in (self.w_, self.V_)]
        self.history_, step = [], 0
        for epoch in range(self.epochs):
            if not len(users):
                break
            order, total_loss = rng.permutation(len(users)), 0.0
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                u, i = users[batch], objects[batch]
                j = self._sample_negatives(u, rng)
                loss, gw, gv = self._bpr_gradients(self._pairs(u, i), self._pairs(u, j))
                step += 1
                for parameter, gradient, (first, second) in zip((self.w_, self.V_), (gw, gv), moments):
                    first *= 0.9
                    first += 0.1 * gradient
                    second *= 0.999
                    second += 0.001 * gradient ** 2
                    parameter -= self.lr * (first / (1 - 0.9 ** step)) / (np.sqrt(second / (1 - 0.999 ** step)) + 1e-8)
                total_loss += loss * len(batch)
            self.history_.append({"epoch": epoch + 1, "bpr_loss": total_loss / len(order)})
        return self

    def predict(self, user_ids, item_ids=None, date=None):
        """Скоры всех train-товаров либо переданных пар; дата не используется."""
        users = self.users_.get_indexer(user_ids)
        if (users < 0).any():
            raise ValueError("Неизвестный пользователь: используйте общий cold-start")
        boundary = len(self.users_)
        user_factors = self.V_[users]
        if item_ids is None:
            features = self.item_features_
        else:
            items = self.items_.get_indexer(item_ids)
            if len(users) != len(items) or (items < 0).any():
                raise ValueError("Неизвестный товар или несовпадение длин ID")
            features = self.item_features_[items]
        item_factors = features @ self.V_[boundary:]
        item_offset = features @ self.w_[boundary:] + 0.5 * np.sum(
            item_factors ** 2 - features.multiply(features) @ (self.V_[boundary:] ** 2), axis=1)
        if item_ids is None:
            return self.w_[users, None] + item_offset[None, :] + user_factors @ item_factors.T
        return self.w_[users] + item_offset + np.sum(user_factors * item_factors, axis=1)
