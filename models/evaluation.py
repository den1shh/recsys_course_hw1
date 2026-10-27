import numpy as np
import pandas as pd


def evaluate(model, targets, seen, k=10, batch_size=256, return_recommendations=False, date=None):
    """Полный train-каталог, исключение истории, NDCG и Recall по пользователям."""
    users = sorted(targets)
    discounts = 1 / np.log2(np.arange(2, k + 2))
    rows, ndcg, recall, supported = [], [], [], []
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        if hasattr(model, "rank"):
            scores, support = model.rank(batch, return_support=True)
            by_support = np.argsort(-support, axis=1, kind="stable")
        else:
            scores = model.predict(batch, date=date)
            by_support = np.broadcast_to(np.arange(len(model.items_)), scores.shape)
        for row, user in enumerate(batch):
            known = model.items_.get_indexer(list(seen.get(user, set())))
            scores[row, known[known >= 0]] = -np.inf
        supported.extend(np.isfinite(scores).any(axis=1).tolist())
        order = np.argsort(-np.take_along_axis(scores, by_support, axis=1), axis=1, kind="stable")[:, :k]
        recommendations = np.take_along_axis(by_support, order, axis=1)
        for row, user in enumerate(batch):
            valid = np.isfinite(scores[row, recommendations[row]])
            item_indices = recommendations[row, valid]
            item_ids = model.items_[item_indices]
            hits = np.array([item in targets[user] for item in item_ids])
            ideal = discounts[:min(k, len(targets[user]))].sum()
            ndcg.append(float((hits * discounts[:len(hits)]).sum() / ideal))
            recall.append(float(hits.sum() / len(targets[user])))
            if return_recommendations:
                rows.extend((user, item, rank, float(scores[row, index]))
                            for rank, (item, index) in enumerate(zip(item_ids, item_indices), 1))
    metrics = {
        f"NDCG@{k}": float(np.mean(ndcg)), f"Recall@{k}": float(np.mean(recall)),
        "Пользователи": len(users), "Есть рекомендации, %": 100 * float(np.mean(supported)),
    }
    if return_recommendations:
        return metrics, pd.DataFrame(rows, columns=["user_id", "item_id", "rank", "score"])
    return metrics
