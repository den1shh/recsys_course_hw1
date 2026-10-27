import numpy as np
import pandas as pd


def recommend_new_users(train, users, train_end, k=10, min_reviews=10, seed=42):
    """Выдачи по доле положительных оценок за последний месяц train."""
    month = train[train["timestamp"] >= train_end - pd.DateOffset(months=1)]
    stats = (
        month.assign(positive=month["rating"] >= 4)
        .groupby("item_id")["positive"].agg(reviews="size", positive_share="mean")
    )
    weights = stats.loc[
        (stats["reviews"] > min_reviews) & (stats["positive_share"] > 0), "positive_share",
    ]
    probabilities = weights / weights.sum()
    rng = np.random.default_rng(seed)
    return {
        user_id: rng.choice(probabilities.index, size=k, replace=False, p=probabilities).tolist()
        for user_id in sorted(set(users))
    }


def insert_new_item(recommendations, new_item, k=10, seed=42):
    """Вставка новинки в выдачи 5% пользователей с весами позиций 1..k."""
    rng = np.random.default_rng(seed)
    users = sorted(recommendations)
    selected = rng.choice(users, size=round(0.05 * len(users)), replace=False)
    ranks = np.arange(1, k + 1)
    positions = rng.choice(k, size=len(selected), p=ranks / ranks.sum())
    result = {user_id: list(items[:k]) for user_id, items in recommendations.items()}

    for user_id, position in zip(selected, positions):
        items = [item for item in result[user_id] if item != new_item]
        result[user_id] = (items[:position] + [new_item] + items[position:])[:k]
    return result
