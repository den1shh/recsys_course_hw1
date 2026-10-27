import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.nn import functional as F


class TimeSVDPlusPlus:
    """SVD++ из семинара 4 с линейным дрейфом Koren (2009), без дневных эффектов.

    r = mu + bu + bi + bi,bin(t) + au*dev + qi @ (pu + apu*dev + H[u] @ y).
    dev = sign(t-tu)*abs(t-tu)**beta; t и tu измеряются в днях UTC.
    Источник: https://yuzhang-teaching.github.io/CSCE670-F25/Koren09.pdf
    """

    def __init__(self, n_factors=16, n_epochs=5, n_bins=30, batch_size=512,
                 lr=0.01, lr_time=0.001, reg=0.02, beta=0.4, random_state=42):
        self.n_factors, self.n_epochs, self.n_bins = n_factors, n_epochs, n_bins
        self.batch_size, self.lr, self.lr_time = batch_size, lr, lr_time
        self.reg, self.beta, self.random_state = reg, beta, random_state

    def fit(self, train):
        self.users_ = pd.Index(sorted(train["user_id"].unique()))
        self.items_ = pd.Index(sorted(train["item_id"].unique()))
        u = self.users_.get_indexer(train["user_id"])
        i = self.items_.get_indexer(train["item_id"])
        n_users, n_items = len(self.users_), len(self.items_)
        times = pd.to_datetime(train["timestamp"], utc=True)
        self.start_date_, self.end_date_ = times.min().normalize(), times.max()
        self.n_days_ = (self.end_date_.normalize() - self.start_date_).days + 1
        days = ((times.dt.normalize() - self.start_date_) / pd.Timedelta(days=1)).to_numpy()
        counts = np.bincount(u, minlength=n_users)
        self.mean_time_ = np.bincount(u, weights=days, minlength=n_users) / counts
        delta = days - self.mean_time_[u]
        dev = (np.sign(delta) * np.abs(delta) ** self.beta).astype(np.float32)
        bins = np.minimum((days * self.n_bins / self.n_days_).astype(int), self.n_bins - 1)
        ratings = train["rating"].to_numpy(np.float32)
        self.global_mean_ = float(ratings.mean())

        history = sparse.csr_matrix((np.ones(len(train), dtype=np.float32), (u, i)),
                                    shape=(n_users, n_items))
        # N(u) — множество товаров, поэтому повторы оценок не умножают y_j.
        history.data = np.repeat(1 / np.sqrt(np.diff(history.indptr)),
                                 np.diff(history.indptr)).astype(np.float32)
        self.history_matrix_ = history
        rng = np.random.default_rng(self.random_state)
        pu = torch.nn.Parameter(torch.tensor(rng.normal(0, 0.05, (n_users, self.n_factors)),
                                             dtype=torch.float32))
        qi = torch.nn.Parameter(torch.tensor(rng.normal(0, 0.05, (n_items, self.n_factors)),
                                             dtype=torch.float32))
        y = torch.nn.Parameter(torch.zeros(n_items, self.n_factors))
        bu = torch.nn.Parameter(torch.zeros(n_users))
        bi = torch.nn.Parameter(torch.zeros(n_items))
        bit = torch.nn.Parameter(torch.zeros(n_items, self.n_bins))
        au = torch.nn.Parameter(torch.zeros(n_users))
        apu = torch.nn.Parameter(torch.zeros(n_users, self.n_factors))
        weights = [pu, qi, y, bu, bi, bit, au, apu]
        optimizer = torch.optim.Adam([
            {"params": [pu, qi, y, bu, bi, bit], "lr": self.lr},
            {"params": [au, apu], "lr": self.lr_time},
        ])

        order = np.argsort(u, kind="stable")
        u, i, bins, dev, ratings = [torch.from_numpy(a[order]) for a in (u, i, bins, dev, ratings)]
        offsets = np.r_[0, np.cumsum(counts)]
        history_items = torch.from_numpy(history.indices.astype(np.int64))
        history_offsets = torch.from_numpy(history.indptr[:-1].astype(np.int64))
        history_weights = torch.from_numpy(history.data)
        self.history_ = []
        for epoch in range(1, self.n_epochs + 1):
            users = rng.permutation(n_users)
            for start in range(0, n_users, self.batch_size):
                batch = users[start:start + self.batch_size]
                rows = np.concatenate([np.arange(offsets[a], offsets[a + 1]) for a in batch])
                local = torch.from_numpy(np.repeat(np.arange(len(batch)), counts[batch]))
                h = history[batch]
                implicit = F.embedding_bag(
                    torch.from_numpy(h.indices.astype(np.int64)), y,
                    torch.from_numpy(h.indptr[:-1].astype(np.int64)),
                    mode="sum", per_sample_weights=torch.from_numpy(h.data),
                )
                ub, ib, db = u[rows], i[rows], dev[rows]
                effective = pu[ub] + apu[ub] * db[:, None] + implicit[local]
                prediction = (self.global_mean_ + bu[ub] + bi[ib] + bit[ib, bins[rows]]
                              + au[ub] * db + (qi[ib] * effective).sum(dim=1))
                # Выборка пользователей даёт несмещённый градиент общего MSE.
                loss = ((prediction - ratings[rows]) ** 2).sum() * n_users / (len(batch) * len(train))
                loss += self.reg * sum(w.square().sum() for w in weights) / len(train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                implicit = F.embedding_bag(history_items, y, history_offsets, mode="sum",
                                           per_sample_weights=history_weights)
                effective = pu[u] + apu[u] * dev[:, None] + implicit[u]
                fitted = (self.global_mean_ + bu[u] + bi[i] + bit[i, bins]
                          + au[u] * dev + (qi[i] * effective).sum(dim=1))
                rmse = float(torch.sqrt(((fitted - ratings) ** 2).mean()))
            self.history_.append({"epoch": epoch, "train_rmse": rmse})

        (self.user_factors_, self.item_factors_, self.history_factors_,
         self.user_bias_, self.item_bias_, self.item_time_bias_,
         self.user_drift_, self.factor_drift_) = [w.detach().numpy() for w in weights]
        return self

    def predict(self, user_ids, item_ids=None, date=None):
        """Скоры на одну дату рекомендации; по умолчанию — конец train."""
        users = self.users_.get_indexer(user_ids)
        if (users < 0).any():
            raise ValueError("Неизвестный пользователь")
        date = self.end_date_ if date is None else pd.to_datetime(date, utc=True)
        if not isinstance(date, pd.Timestamp):
            raise ValueError("Нужна единая дата рекомендации")
        day = (date.normalize() - self.start_date_).days
        delta = day - self.mean_time_[users]
        dev = (np.sign(delta) * np.abs(delta) ** self.beta).astype(np.float32)
        # За пределами train используем ближайший крайний интервал товара.
        bin_id = int(np.clip(day * self.n_bins // self.n_days_, 0, self.n_bins - 1))
        effective = (self.user_factors_[users] + self.factor_drift_[users] * dev[:, None]
                     + self.history_matrix_[users] @ self.history_factors_)
        baseline = self.global_mean_ + self.user_bias_[users] + self.user_drift_[users] * dev
        item_bias = self.item_bias_ + self.item_time_bias_[:, bin_id]
        if item_ids is None:
            scores = effective @ self.item_factors_.T
            scores += baseline[:, None]
            scores += item_bias[None, :]
            return scores
        items = self.items_.get_indexer(item_ids)
        if len(users) != len(items) or (items < 0).any():
            raise ValueError("Неизвестный товар или несовпадение длин ID")
        return baseline + item_bias[items] + (effective * self.item_factors_[items]).sum(axis=1)
