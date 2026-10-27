from itertools import product
from time import perf_counter

import pandas as pd

from .evaluation import evaluate
from .neighborhood import NeighborhoodCF


def tune_neighbors(train, validation, k_values=(20, 40, 80), results_path=None):
    """Сначала сравниваем варианты CF, затем уточняем параметры лидера."""
    rows = []

    def run(kind, similarity, min_common, shrinkage):
        started = perf_counter()
        model = NeighborhoodCF(kind, similarity, max(k_values), min_common, shrinkage).fit(train)
        for k in k_values:
            metrics = evaluate(model.set_neighbors(k), validation["targets"], validation["seen"])
            rows.append({
                "kind": kind, "similarity": similarity, "n_neighbors": k,
                "min_common": min_common, "shrinkage": shrinkage, **metrics,
            })
        best = max(row["NDCG@10"] for row in rows[-len(k_values):])
        print(f"{kind}, {similarity}, min_common={min_common}, shrinkage={shrinkage}: "
              f"NDCG@10={best:.4f}, {perf_counter() - started:.1f} с", flush=True)
        if results_path is not None:
            pd.DataFrame(rows).to_csv(results_path, index=False)

    similarities = ["cosine", "centered_cosine", "binary_cosine", "jaccard"]
    for kind, similarity in product(["user", "item"], similarities):
        run(kind, similarity, min_common=3, shrinkage=10)
    leader = max(rows, key=lambda row: (row["NDCG@10"], row["Recall@10"]))
    for min_common, shrinkage in product([1, 3, 5], [0, 10, 50]):
        if (min_common, shrinkage) != (3, 10):
            run(leader["kind"], leader["similarity"], min_common, shrinkage)

    results = pd.DataFrame(rows).sort_values(
        ["NDCG@10", "Recall@10", "n_neighbors"], ascending=[False, False, True], kind="stable",
    ).reset_index(drop=True)
    parameter_names = ["kind", "similarity", "n_neighbors", "min_common", "shrinkage"]
    parameters = results.iloc[0][parameter_names].to_dict()
    for name in parameter_names[2:]:
        parameters[name] = int(parameters[name])
    if results_path is not None:
        results.to_csv(results_path, index=False)
    return results, parameters
