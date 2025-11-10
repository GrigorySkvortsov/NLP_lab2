#!/usr/bin/env python3
"""
stage2_vectorization.py

Этап 2. Реализация классических методов векторизации с перебором параметров.

Что делает:
- Загружает corpus.jsonl (ожидает поле "text")
- Выполняет векторизацию по сетке параметров:
  - ngram_range: (1,1), (1,2), (1,3)
  - методы: onehot (binary Count), bow (Count), tfidf (Tfidf)
  - Для TF-IDF перебираются комбинации параметров:
      smooth_idf: True/False
      sublinear_tf: True/False
- Для каждого конфигурационного кейса сохраняет:
  - разрежённую матрицу в data/{config_name}_matrix.npz
  - соответствующий vectorizer в data/{config_name}_vectorizer.pkl
  - метрики в data/vectorization_metrics.csv (одна строка = один кейс)
- Также сохраняет общий meta JSON с перечнем кейсов.

Запуск:
    python stage2_vectorization.py

Зависимости:
    pip install scikit-learn scipy pandas joblib
"""

import os
import json
import pickle
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ---------- Настройки (при необходимости изменяй) ----------
CORPUS_PATH = "input/corpus.jsonl"
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
MAX_FEATURES = 50000
NGRAM_RANGES = [(1, 1), (1, 2), (1, 3)]
METHODS = ["onehot", "bow", "tfidf"]
TFIDF_PARAM_GRID = list(product([True, False], [True, False]))  # (smooth_idf, sublinear_tf)
# -----------------------------------------------------------

def load_corpus_texts(path):
    texts = []
    docs_meta = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj.get("text", ""))
            docs_meta.append(obj)  # keep full object for possible labels
    return texts, docs_meta

def estimate_sparse_memory_bytes(matrix):
    # estimate memory used by CSR/CSC arrays
    if sp.issparse(matrix):
        return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
    return 0

def make_config_name(method, ngram_range, tfidf_params=None):
    ng = f"ng{ngram_range[0]}-{ngram_range[1]}"
    if method == "tfidf" and tfidf_params:
        s = f"smooth{tfidf_params['smooth_idf']}_sublinear{tfidf_params['sublinear_tf']}"
        return f"tfidf_{s}_{ng}"
    else:
        return f"{method}_{ng}"

def build_vectorizer(method, ngram_range, tfidf_params=None):
    if method == "onehot":
        return CountVectorizer(binary=True, max_features=MAX_FEATURES, ngram_range=ngram_range)
    if method == "bow":
        return CountVectorizer(binary=False, max_features=MAX_FEATURES, ngram_range=ngram_range)
    if method == "tfidf":
        params = {"max_features": MAX_FEATURES, "ngram_range": ngram_range}
        if tfidf_params:
            params.update({"smooth_idf": tfidf_params["smooth_idf"], "sublinear_tf": tfidf_params["sublinear_tf"]})
        return TfidfVectorizer(**params)
    raise ValueError("Unknown method")

def main():
    print("Loading corpus...")
    texts, docs_meta = load_corpus_texts(CORPUS_PATH)
    n_docs = len(texts)
    print(f"Loaded {n_docs} documents.")

    all_results = []
    meta_list = []

    # iterate through parameter grid
    for method in METHODS:
        for ngram_range in NGRAM_RANGES:
            if method == "tfidf":
                for smooth_idf, sublinear_tf in TFIDF_PARAM_GRID:
                    tfidf_params = {"smooth_idf": smooth_idf, "sublinear_tf": sublinear_tf}
                    cfg_name = make_config_name(method, ngram_range, tfidf_params)
                    print(f"\n-> Fitting {cfg_name}")
                    vec = build_vectorizer(method, ngram_range, tfidf_params)
                    X = vec.fit_transform(texts)
                    nnz = X.nnz
                    n_docs, n_features = X.shape
                    density = nnz / (n_docs * n_features) if n_docs * n_features > 0 else 0.0
                    sparsity = 1.0 - density
                    mem_bytes = estimate_sparse_memory_bytes(X)

                    # save
                    sp.save_npz(DATA_DIR / f"{cfg_name}_matrix.npz", X)
                    with open(DATA_DIR / f"{cfg_name}_vectorizer.pkl", "wb") as f:
                        pickle.dump(vec, f)

                    row = {
                        "config": cfg_name,
                        "method": method,
                        "ngram_min": ngram_range[0],
                        "ngram_max": ngram_range[1],
                        "tfidf_smooth_idf": smooth_idf,
                        "tfidf_sublinear_tf": sublinear_tf,
                        "n_documents": int(n_docs),
                        "n_features": int(n_features),
                        "nnz": int(nnz),
                        "density": float(density),
                        "sparsity": float(sparsity),
                        "memory_bytes": int(mem_bytes),
                        "max_features": MAX_FEATURES
                    }
                    all_results.append(row)
                    meta_list.append(row)
            else:
                cfg_name = make_config_name(method, ngram_range, None)
                print(f"\n-> Fitting {cfg_name}")
                vec = build_vectorizer(method, ngram_range, None)
                X = vec.fit_transform(texts)
                nnz = X.nnz
                n_docs, n_features = X.shape
                density = nnz / (n_docs * n_features) if n_docs * n_features > 0 else 0.0
                sparsity = 1.0 - density
                mem_bytes = estimate_sparse_memory_bytes(X)

                # save
                sp.save_npz(DATA_DIR / f"{cfg_name}_matrix.npz", X)
                with open(DATA_DIR / f"{cfg_name}_vectorizer.pkl", "wb") as f:
                    pickle.dump(vec, f)

                row = {
                    "config": cfg_name,
                    "method": method,
                    "ngram_min": ngram_range[0],
                    "ngram_max": ngram_range[1],
                    "tfidf_smooth_idf": None,
                    "tfidf_sublinear_tf": None,
                    "n_documents": int(n_docs),
                    "n_features": int(n_features),
                    "nnz": int(nnz),
                    "density": float(density),
                    "sparsity": float(sparsity),
                    "memory_bytes": int(mem_bytes),
                    "max_features": MAX_FEATURES
                }
                all_results.append(row)
                meta_list.append(row)

    # save overall CSV and meta JSON
    df = pd.DataFrame(all_results)
    df.to_csv(DATA_DIR / "vectorization_metrics.csv", index=False)
    with open(DATA_DIR / "vectorization_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)

    # save docs_meta (for potential label-based evaluation later)
    with open(DATA_DIR / "docs_meta.jsonl", "w", encoding="utf-8") as f:
        for obj in docs_meta:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("\n✅ Stage 2 finished. All matrices and vectorizers saved to 'data/'.")
    print(f"Saved {len(all_results)} configurations.")
    print("Vectorization metrics CSV: data/vectorization_metrics.csv")

if __name__ == "__main__":
    main()
