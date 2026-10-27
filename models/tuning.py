from time import perf_counter

import pandas as pd

from .evaluation import evaluate


def tune_model(model_class, configurations, train, validation, items=None, k=10, results_path=None):
    """Небольшой список настроек; выбираем по NDCG, затем по Recall."""
    rows, best_model, best_key = [], None, (-1, -1)
    for parameters in configurations:
        model = model_class(**parameters)
        started = perf_counter()
        model.fit(train) if items is None else model.fit(train, items)
        fit_seconds = perf_counter() - started
        started = perf_counter()
        metrics = evaluate(model, validation["targets"], validation["seen"], k=k,
                           date=validation["date"])
        rows.append({"parameters": parameters, **metrics, "fit, с": fit_seconds,
                     "validation, с": perf_counter() - started})
        key = metrics[f"NDCG@{k}"], metrics[f"Recall@{k}"]
        if key > best_key:
            best_model, best_key = model, key
        print(f"{model_class.__name__} {parameters}: NDCG@{k}={key[0]:.5f}, "
              f"{fit_seconds + rows[-1]['validation, с']:.1f} с", flush=True)
        if results_path is not None:
            pd.DataFrame(rows).to_csv(results_path, index=False)
    results = pd.DataFrame(rows).sort_values(
        [f"NDCG@{k}", f"Recall@{k}"], ascending=False, kind="stable",
    ).reset_index(drop=True)
    return best_model, results
