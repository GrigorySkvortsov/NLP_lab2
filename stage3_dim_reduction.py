#!/usr/bin/env python3
"""
stage3_dim_reduction.py

Этап 3. Снижение размерности и тематическое моделирование (LSA).
Этот скрипт загружает результаты Stage 2 (data/*_matrix.npz и vectorizers) и для каждой конфигурации:
- применяет TruncatedSVD для ряда значений n_components
- сохраняет объяснённую дисперсию по каждому n_components
- сохраняет топ-термины для компонент (тематическая интерпретация)
- визуализирует зависимость объяснённой дисперсии от числа компонент
- строит 2D визуализации (UMAP если доступен, иначе TSNE)
- при наличии метки category сравнивает KMeans-кластеризацию с истинными метками (ARI)

Запуск:
    python stage3_dim_reduction.py

Зависимости:
    pip install scikit-learn scipy pandas matplotlib seaborn umap-learn
    (umap-learn опционален; если отсутствует — будет использоваться TSNE)
"""

import os
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

# Try to import UMAP if available
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

# ---------- Настройки ----------
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "topic_terms").mkdir(exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)

N_COMPONENTS_LIST = [10, 50, 100, 200, 300]  # исследуемые числа компонент
TOP_TERMS = 15
# -----------------------------------------------------------

def load_meta():
    meta_path = DATA_DIR / "vectorization_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("data/vectorization_meta.json not found — запустите stage2_vectorization.py")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta

def load_docs_meta():
    docs_path = DATA_DIR / "docs_meta.jsonl"
    if not docs_path.exists():
        return None
    docs = []
    with open(docs_path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs

def safe_tsne_transform(X_reduced):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="random", learning_rate="auto")
    return tsne.fit_transform(X_reduced)

def safe_umap_transform(X_reduced):
    if UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, random_state=42)
        return reducer.fit_transform(X_reduced)
    else:
        return None

def analyze_config(cfg):
    cfg_name = cfg["config"]
    matrix_path = DATA_DIR / f"{cfg_name}_matrix.npz"
    vectorizer_path = DATA_DIR / f"{cfg_name}_vectorizer.pkl"
    if not matrix_path.exists() or not vectorizer_path.exists():
        print(f"Skipping {cfg_name}: missing files.")
        return None

    print(f"\n=== Analyzing {cfg_name} ===")
    X = sp.load_npz(matrix_path)
    with open(vectorizer_path, "rb") as f:
        vec = pickle.load(f)
    try:
        terms = np.array(vec.get_feature_names_out())
    except Exception:
        terms = None

    n_docs, n_feats = X.shape

    per_cfg_results = {"config": cfg_name, "n_features": n_feats, "n_documents": n_docs, "svd": []}

    for k in N_COMPONENTS_LIST:
        print(f"  - computing SVD (k={k}) ...")
        svd = TruncatedSVD(n_components=k, random_state=42)
        X_red = svd.fit_transform(X)

        explained = float(svd.explained_variance_ratio_.sum())
        per_cfg_results["svd"].append({"n_components": k, "explained_variance": explained})

        # save top terms for first components
        if terms is not None:
            top_terms_list = []
            for i, comp in enumerate(svd.components_[:min(10, k)]):
                top_idx = np.argsort(comp)[::-1][:TOP_TERMS]
                top_terms = [terms[j] for j in top_idx]
                top_terms_list.append(top_terms)
            # save into file per config and k
            out_topics = OUTPUT_DIR / "topic_terms" / f"{cfg_name}_k{k}_topics.txt"
            with open(out_topics, "w", encoding="utf-8") as f:
                for idx, tt in enumerate(top_terms_list, start=1):
                    f.write(f"Component {idx}: {', '.join(tt)}\n")

        # 2D visualization (UMAP if available, else TSNE) using X_red
        umap_2d = safe_umap_transform(X_red) if UMAP_AVAILABLE else None
        if umap_2d is None:
            try:
                tsne_2d = safe_tsne_transform(X_red)
            except Exception as e:
                print("    Warning: TSNE failed:", e)
                tsne_2d = None
        else:
            tsne_2d = None

        if umap_2d is not None:
            vis = umap_2d
            alg = "umap"
        elif tsne_2d is not None:
            vis = tsne_2d
            alg = "tsne"
        else:
            vis = None
            alg = None

        if vis is not None:
            plt.figure(figsize=(7,5))
            sns.scatterplot(x=vis[:,0], y=vis[:,1], s=10)
            plt.title(f"{cfg_name} — {k} comps — {alg}")
            plt.xlabel("dim1")
            plt.ylabel("dim2")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "figures" / f"{cfg_name}_k{k}_{alg}.png")
            plt.close()

        # clustering-based quality check (if labels exist)
        docs_meta = load_docs_meta()
        if docs_meta is not None:
            # collect categories if available
            cats = [d.get("category") for d in docs_meta]
            if all(c is not None for c in cats):
                # encode categories to integers
                uniq = sorted(set(cats))
                y_true = np.array([uniq.index(c) for c in cats])
                # run KMeans with num clusters = number of unique categories (cap to reasonable)
                n_clusters = min(len(uniq), 20)  # cap to 20 for robustness
                try:
                    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    y_pred = km.fit_predict(X_red)
                    # ARI is defined even with different label sets
                    ari = adjusted_rand_score(y_true, y_pred)
                    # silhouette (needs >1 cluster and <n_samples)
                    sil = silhouette_score(X_red, y_pred) if 1 < n_clusters < len(y_true) else None
                    per_cfg_results["svd"][-1].update({"ari": float(ari), "silhouette": float(sil) if sil is not None else None})
                except Exception as e:
                    per_cfg_results["svd"][-1].update({"ari": None, "silhouette": None, "clustering_error": str(e)})
            else:
                per_cfg_results["svd"][-1].update({"ari": None, "silhouette": None})
        else:
            per_cfg_results["svd"][-1].update({"ari": None, "silhouette": None})

    return per_cfg_results

def main():
    meta = load_meta()
    overall = []
    for cfg in meta:
        res = analyze_config(cfg)
        if res is not None:
            overall.append(res)

    # flatten results to table
    rows = []
    for cfg_res in overall:
        cfg_name = cfg_res["config"]
        for s in cfg_res["svd"]:
            rows.append({
                "config": cfg_name,
                "n_features": cfg_res["n_features"],
                "n_documents": cfg_res["n_documents"],
                "n_components": s["n_components"],
                "explained_variance": s.get("explained_variance"),
                "ari": s.get("ari"),
                "silhouette": s.get("silhouette")
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "lsa_experiments.csv", index=False)

    # Plot explained variance curves for each config (saved)
    configs = df["config"].unique()
    plt.figure(figsize=(10,6))
    for cfg in configs:
        sub = df[df["config"] == cfg]
        plt.plot(sub["n_components"], sub["explained_variance"], marker="o", label=cfg)
    plt.xlabel("n_components")
    plt.ylabel("explained_variance")
    plt.title("Explained variance vs n_components for different vectorization configs")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "figures" / "explained_variance_all_configs.png")
    plt.close()

    print("\n✅ Stage 3 finished. Results saved to outputs/ (lsa_experiments.csv, topic_terms/, figures/).")

if __name__ == "__main__":
    main()
