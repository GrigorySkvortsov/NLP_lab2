#!/usr/bin/env python3
"""
stage4_vectorization_analysis.py

Этап 4. Сравнительный анализ классических методов векторизации.

Использует сохранённые данные из этапа 2 (матрицы, метаданные и корпус).

Проводит:
- анализ размерности, разреженности, памяти;
- вычисление семантической согласованности (косинусное сходство внутри темы);
- оценку вычислительной эффективности (время векторизации);
- создание сводной таблицы и визуализаций.

Запуск:
    python stage4_vectorization_analysis.py

Зависимости:
    pip install scikit-learn scipy pandas matplotlib seaborn
"""

import os
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- вспомогательные функции ---
def load_meta():
    path = DATA_DIR / "vectorization_meta.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_docs_meta():
    path = DATA_DIR / "docs_meta.jsonl"
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs

def estimate_sparse_memory_bytes(matrix):
    if sp.issparse(matrix):
        return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    return matrix.nbytes

def compute_semantic_coherence(X, docs_meta):
    """
    Вычисляет среднее косинусное сходство между документами одной темы (category).
    """
    if docs_meta is None or not all("category" in d for d in docs_meta):
        return None

    categories = [d["category"] for d in docs_meta]
    uniq = list(sorted(set(categories)))
    coherence_scores = []

    X_dense = X.toarray() if sp.issparse(X) and X.shape[0] < 2000 else None
    # (для крупных корпусов можно ограничить выборкой)
    for cat in uniq:
        idx = [i for i, c in enumerate(categories) if c == cat]
        if len(idx) < 2:
            continue
        X_sub = X[idx] if X_dense is None else X_dense[idx]
        sims = cosine_similarity(X_sub)
        upper = np.triu_indices_from(sims, k=1)
        coherence_scores.append(float(np.mean(sims[upper])))
    if len(coherence_scores) == 0:
        return None
    return float(np.mean(coherence_scores))

# --- основной код ---
def main():
    meta = load_meta()
    docs_meta = load_docs_meta()

    print(f"Loaded {len(meta)} vectorization configs.")
    print(f"Docs meta loaded: {len(docs_meta)} entries")

    results = []

    for cfg in meta:
        cfg_name = cfg["config"]
        matrix_path = DATA_DIR / f"{cfg_name}_matrix.npz"
        vectorizer_path = DATA_DIR / f"{cfg_name}_vectorizer.pkl"

        if not matrix_path.exists():
            print(f"⚠️  Skip {cfg_name}: matrix file not found.")
            continue

        print(f"\nAnalyzing {cfg_name}...")
        start_time = time.time()
        X = sp.load_npz(matrix_path)
        elapsed_load = time.time() - start_time

        # базовые метрики
        n_docs, n_features = X.shape
        nnz = X.nnz
        density = nnz / (n_docs * n_features)
        sparsity = 1.0 - density
        mem_bytes = estimate_sparse_memory_bytes(X)

        # семантическая согласованность
        sample_docs = docs_meta
        if sample_docs and len(sample_docs) == n_docs:
            coherence = compute_semantic_coherence(X, sample_docs)
        else:
            coherence = None

        # итоговая запись
        row = {
            "config": cfg_name,
            "method": cfg["method"],
            "ngram_min": cfg["ngram_min"],
            "ngram_max": cfg["ngram_max"],
            "tfidf_smooth_idf": cfg.get("tfidf_smooth_idf"),
            "tfidf_sublinear_tf": cfg.get("tfidf_sublinear_tf"),
            "n_documents": n_docs,
            "n_features": n_features,
            "nnz": nnz,
            "density": density,
            "sparsity": sparsity,
            "memory_bytes": mem_bytes,
            "load_time_sec": elapsed_load,
            "semantic_coherence": coherence,
        }
        results.append(row)

    df = pd.DataFrame(results)
    out_csv = DATA_DIR / "vectorization_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✅ Results saved to {out_csv}")

    # --- визуализации ---
    sns.set_theme(style="whitegrid")

    # 1. Размерность и плотность
    plt.figure(figsize=(8,5))
    sns.barplot(data=df, x="method", y="n_features", hue="ngram_max")
    plt.title("Размерность векторного пространства по методам и n-граммам")
    plt.ylabel("Количество признаков")
    plt.xlabel("Метод векторизации")
    plt.legend(title="N-граммы")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "vectorization_dimensionality.png")
    plt.close()

    # 2. Разреженность
    plt.figure(figsize=(8,5))
    sns.barplot(data=df, x="method", y="density", hue="ngram_max")
    plt.title("Плотность (1 - разреженность) матриц по методам")
    plt.ylabel("Доля ненулевых элементов")
    plt.xlabel("Метод")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "vectorization_density.png")
    plt.close()

    # 3. Семантическая согласованность
    if df["semantic_coherence"].notnull().any():
        plt.figure(figsize=(8,5))
        sns.barplot(data=df, x="method", y="semantic_coherence", hue="ngram_max")
        plt.title("Семантическая согласованность (внутри категорий)")
        plt.ylabel("Среднее косинусное сходство")
        plt.xlabel("Метод")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "semantic_coherence.png")
        plt.close()

    print("✅ Plots saved to outputs/")

if __name__ == "__main__":
    main()
