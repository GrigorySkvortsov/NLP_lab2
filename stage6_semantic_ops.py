#!/usr/bin/env python3
"""
stage6_semantic_ops.py

Этап 6. Эксперименты с векторной арифметикой и семантическими операциями.

Использует модели из data/models/*.model (gensim Word2Vec/FastText/Doc2Vec)
и (опционально) файлы:
 - data/analogies.txt         (word2vec-format analogies)
 - data/test_words.txt        (список слов для матриц и соседей)
 - data/word_similarity.csv   (word1,word2,human_score) -- для корреляции
 - data/synonyms.csv          (word1,word2) -- пары синонимов
 - data/antonyms.csv          (word1,word2) -- пары антонимов
 - data/morph_pairs.csv       (word1,word2) -- морфологические примеры

Сохраняет результаты в outputs/semantic_ops/

Примеры вызова:
    python stage6_semantic_ops.py
    python stage6_semantic_ops.py --models data/models --out outputs/semantic_ops --topn 10 --sample_pairs 2000

Зависимости:
    pip install gensim numpy scipy pandas scikit-learn matplotlib
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from gensim.models import KeyedVectors, Word2Vec, FastText, Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform

# --------------------- Настройки по умолчанию ---------------------
DEFAULT_MODELS_DIR = Path("data/models")
DEFAULT_OUT = Path("outputs/semantic_ops")
DEFAULT_SAMPLE_PAIRS = 3000  # количество пар для оценки распределения расстояний
DEFAULT_TOPN = 10
DEFAULT_TEST_WORDS = Path("data/test_words.txt")
ANALOGIES_PATH = Path("data/analogies.txt")
WORD_SIM_PATH = Path("data/word_similarity.csv")
SYNONYMS_PATH = Path("data/synonyms.csv")
ANTONYMS_PATH = Path("data/antonyms.csv")
MORPH_PATH = Path("data/morph_pairs.csv")
# -----------------------------------------------------------------

# --- Полезные вспомогательные функции ---
def list_model_files(models_dir: Path) -> List[Path]:
    if not models_dir.exists():
        return []
    # принимаем .model и .kv (gensim)
    files = sorted(models_dir.glob("*.model")) + sorted(models_dir.glob("*.kv")) + sorted(models_dir.glob("*.bin"))
    return files

def load_gensim_model(path: Path):
    # пытаемся загрузить как gensim model (Word2Vec/FastText/Doc2Vec) или KeyedVectors
    try:
        m = Word2Vec.load(str(path))
        return m
    except Exception:
        try:
            m = FastText.load(str(path))
            return m
        except Exception:
            try:
                m = Doc2Vec.load(str(path))
                return m
            except Exception:
                try:
                    kv = KeyedVectors.load(str(path))
                    return kv
                except Exception:
                    # try load_word2vec_format (binary)
                    try:
                        kv = KeyedVectors.load_word2vec_format(str(path), binary=True)
                        return kv
                    except Exception as e:
                        print(f"  Failed to load model {path}: {e}")
                        return None

def get_kv(model):
    # возвращает KeyedVectors-подобный объект с векторными операциями
    if model is None:
        return None
    if hasattr(model, "wv"):
        return model.wv
    # Doc2Vec: имеет .wv (in gensim>=4 doc2vec also has .wv?), but doc vectors are model.dv
    if isinstance(model, Doc2Vec):
        return model.wv  # for word vectors
    if isinstance(model, KeyedVectors):
        return model
    return None

def safe_most_similar(kv, positive, negative=None, topn=10):
    negative = negative or []
    try:
        return kv.most_similar(positive=positive, negative=negative, topn=topn)
    except Exception:
        # fallback: compute manually
        vec = np.zeros(kv.vector_size, dtype=np.float32)
        for p in positive:
            vec += kv[p]
        for n in negative:
            vec -= kv[n]
        all_words = kv.index_to_key if hasattr(kv, "index_to_key") else list(kv.key_to_index.keys())
        sims = []
        for w in all_words:
            try:
                sims.append((w, float(np.dot(vec, kv[w]) / (np.linalg.norm(vec) * np.linalg.norm(kv[w]) + 1e-12))))
            except Exception:
                continue
        sims.sort(key=lambda x: -x[1])
        return sims[:topn]

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0

# ----------------- Основные экспериментальные процедуры -----------------

def sample_pairwise_similarities(kv, sample_pairs: int = 3000) -> np.ndarray:
    """
    Вычисляет распределение косинусных сходств между случайными парами слов из словаря.
    Возвращает массив similarity значений.
    """
    words = list(kv.index_to_key) if hasattr(kv, "index_to_key") else list(kv.key_to_index.keys())
    n = len(words)
    if n < 2:
        return np.array([])
    sims = []
    for _ in range(sample_pairs):
        a, b = random.randrange(n), random.randrange(n)
        if a == b:
            continue
        wa, wb = words[a], words[b]
        try:
            sims.append(kv.similarity(wa, wb))
        except Exception:
            # fallback compute manually
            sims.append(cosine(kv[wa], kv[wb]))
    return np.array(sims)

def build_similarity_matrix_for_test_words(kv, test_words: List[str], topn=50) -> Dict[str, List[Tuple[str, float]]]:
    """
    Для каждого слова из test_words возвращает topn ближайших слов и их сходство.
    """
    res = {}
    for w in test_words:
        if w in kv.key_to_index:
            try:
                res[w] = kv.most_similar(w, topn=topn)
            except Exception:
                # fallback
                res[w] = safe_most_similar(kv, positive=[w], topn=topn)
        else:
            res[w] = []
    return res

def evaluate_synonym_antonym_pairs(kv, pairs_path: Path) -> Dict[str, Any]:
    """
    Для каждой пары (word1, word2) возвращает cosine. Возвращает средние значения.
    """
    if not pairs_path.exists():
        return None
    df = pd.read_csv(pairs_path, header=None, names=["w1","w2"])
    scores = []
    coverage = 0
    total = len(df)
    for idx, row in df.iterrows():
        w1, w2 = str(row["w1"]), str(row["w2"])
        if w1 in kv.key_to_index and w2 in kv.key_to_index:
            s = kv.similarity(w1, w2)
            scores.append(s)
            coverage += 1
    return {
        "mean_similarity": float(np.mean(scores)) if len(scores) else None,
        "median_similarity": float(np.median(scores)) if len(scores) else None,
        "coverage": coverage,
        "total_pairs": total
    }

def evaluate_analogies_topk(kv, analogies_path: Path, topk_list: List[int] = [1,5]) -> Dict[str, Any]:
    """
    Простая реализация оценки аналогий: файл аналогий должен содержать строки: a b c d
    (a:b :: c:d). Для каждой строки проверяем, находится ли d среди top-k результатов для b - a + c.
    Возвращаем accuracy для каждого k.
    """
    if not analogies_path.exists():
        return None
    with open(analogies_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
    counts = {k: 0 for k in topk_list}
    total = 0
    for ln in lines:
        toks = ln.split()
        if len(toks) < 4:
            continue
        a, b, c, d = toks[0], toks[1], toks[2], toks[3]
        if any(w not in kv.key_to_index for w in [a,b,c,d]):
            continue
        total += 1
        try:
            res = kv.most_similar(positive=[b, c], negative=[a], topn=max(topk_list))
            preds = [r[0] for r in res]
        except Exception:
            # fallback manual
            vec = kv[b] - kv[a] + kv[c]
            all_words = kv.index_to_key
            sims = []
            for w in all_words:
                sims.append((w, float(np.dot(vec, kv[w])/(np.linalg.norm(vec)*np.linalg.norm(kv[w])+1e-12))))
            sims.sort(key=lambda x: -x[1])
            preds = [w for w,_ in sims[:max(topk_list)]]
        for k in topk_list:
            if d in preds[:k]:
                counts[k] += 1
    if total == 0:
        return None
    acc = {f"top{k}_acc": counts[k] / total for k in topk_list}
    acc.update({"evaluated_pairs": total})
    return acc

def compute_axis_from_pairs(kv, pairs: List[Tuple[str,str]]) -> np.ndarray:
    """
    Строит ось (вектор) как средняя разности vec(a)-vec(b) для списка пар (a,b)
    (например: ('мужчина','женщина'), ...)
    """
    vecs = []
    for a,b in pairs:
        if a in kv.key_to_index and b in kv.key_to_index:
            vecs.append(kv[a] - kv[b])
    if not vecs:
        return None
    return np.mean(vecs, axis=0)

def project_words_on_axis(kv, axis: np.ndarray, words: List[str], topn=20) -> Dict[str, float]:
    if axis is None:
        return {}
    out = {}
    axis_norm = axis / (np.linalg.norm(axis) + 1e-12)
    for w in words:
        if w in kv.key_to_index:
            out[w] = float(np.dot(kv[w], axis_norm))
    # return sorted by value
    return dict(sorted(out.items(), key=lambda x: -x[1]))

def nearest_neighbors_and_coherence(kv, words: List[str], topn=10) -> Dict[str, Any]:
    """
    Для каждого слова возвращает topn соседей и метрику coherence:
      coherence = mean(cos(word, neighbor)) и averaged pairwise similarity among neighbors.
    """
    res = {}
    for w in words:
        if w not in kv.key_to_index:
            res[w] = {"neighbors": [], "coherence_word_neighbors": None, "coherence_neighbors_pairwise": None}
            continue
        neigh = kv.most_similar(w, topn=topn)
        neigh_words = [t[0] for t in neigh]
        sims_word_neighbors = [kv.similarity(w, nw) for nw in neigh_words]
        # pairwise among neighbors
        pairwise = []
        for i in range(len(neigh_words)):
            for j in range(i+1, len(neigh_words)):
                pairwise.append(kv.similarity(neigh_words[i], neigh_words[j]))
        res[w] = {
            "neighbors": neigh,
            "coherence_word_neighbors": float(np.mean(sims_word_neighbors)) if sims_word_neighbors else None,
            "coherence_neighbors_pairwise": float(np.mean(pairwise)) if pairwise else None
        }
    return res

# ---------------- Main runner ----------------

def run_experiments(models_dir: Path,
                    out_dir: Path,
                    topn: int = 10,
                    sample_pairs: int = 3000,
                    test_words_path: Path = DEFAULT_TEST_WORDS):
    out_dir.mkdir(parents=True, exist_ok=True)
    model_files = list_model_files(models_dir)
    if not model_files:
        print(f"No models found in {models_dir}. Run stage5 first.")
        return

    # optional test words
    test_words = []
    if test_words_path.exists():
        with open(test_words_path, "r", encoding="utf-8") as f:
            test_words = [l.strip() for l in f if l.strip()]
    else:
        # default small list useful for news
        test_words = ["путин","москва","россия","экономика","спорт","футбол","война","город","работа","слово","король","королева","машина","компания"]

    overall_results = []

    for model_path in model_files:
        print(f"\n--- Processing model {model_path.name} ---")
        model = load_gensim_model(model_path)
        if model is None:
            print("  failed to load, skipping.")
            continue
        kv = get_kv(model)
        if kv is None:
            print("  no keyed vectors, skipping.")
            continue

        model_name = model_path.stem
        model_out = out_dir / model_name
        model_out.mkdir(exist_ok=True)

        # 1) distribution of pairwise similarities (sampled)
        sims = sample_pairwise_similarities(kv, sample_pairs=sample_pairs)
        # save histogram plot
        plt.figure(figsize=(6,4))
        plt.hist(sims, bins=60)
        plt.title(f"Cosine similarity distribution — {model_name}")
        plt.xlabel("cosine")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(model_out / f"{model_name}_cosine_hist.png")
        plt.close()

        # 2) similarity matrix (nearest neighbors) for test words
        sim_matrix = build_similarity_matrix_for_test_words(kv, test_words, topn=topn)
        with open(model_out / f"{model_name}_test_words_neighbors.json", "w", encoding="utf-8") as f:
            json.dump(sim_matrix, f, ensure_ascii=False, indent=2)

        # 3) synonyms / antonyms evaluation (if files exist)
        syn_res = evaluate_synonym_antonym_pairs(kv, SYNONYMS_PATH) if SYNONYMS_PATH.exists() else None
        ant_res = evaluate_synonym_antonym_pairs(kv, ANTONYMS_PATH) if ANTONYMS_PATH.exists() else None
        morph_res = evaluate_synonym_antonym_pairs(kv, MORPH_PATH) if MORPH_PATH.exists() else None

        # 4) analogies evaluation top1/top5 using data/analogies.txt
        analogies_res = evaluate_analogies_topk(kv, ANALOGIES_PATH, topk_list=[1,5]) if ANALOGIES_PATH.exists() else None

        # 5) compute projections on semantic axes (gender, professional, evaluative)
        # default pairs (you can extend or replace by files)
        gender_pairs = [("мужчина","женщина"), ("он","она"), ("сын","дочь"), ("отец","мать"), ("герой","героиня")]
        prof_pairs = [("врач","медсестра"), ("актёр","актриса"), ("учитель","учительница"), ("повар","повариха")]  # example
        eval_pairs = [("хороший","плохой"), ("богатый","бедный"), ("счастливый","несчастный")]

        gender_axis = compute_axis_from_pairs(kv, gender_pairs)
        prof_axis = compute_axis_from_pairs(kv, prof_pairs)
        eval_axis = compute_axis_from_pairs(kv, eval_pairs)

        # project top vocab words or test_words onto axes (we'll use test_words + top freq words)
        # take top 500 words from vocab (or all if small)
        vocab_words = kv.index_to_key[:500] if hasattr(kv, "index_to_key") else list(kv.key_to_index.keys())[:500]
        projection_words = list(dict.fromkeys(test_words + vocab_words))  # keep order, unique

        gender_proj = project_words_on_axis(kv, gender_axis, projection_words, topn=50)
        prof_proj = project_words_on_axis(kv, prof_axis, projection_words, topn=50)
        eval_proj = project_words_on_axis(kv, eval_axis, projection_words, topn=50)

        # save projections
        with open(model_out / f"{model_name}_axis_gender.json", "w", encoding="utf-8") as f:
            json.dump(gender_proj, f, ensure_ascii=False, indent=2)
        with open(model_out / f"{model_name}_axis_prof.json", "w", encoding="utf-8") as f:
            json.dump(prof_proj, f, ensure_ascii=False, indent=2)
        with open(model_out / f"{model_name}_axis_eval.json", "w", encoding="utf-8") as f:
            json.dump(eval_proj, f, ensure_ascii=False, indent=2)

        # 6) nearest neighbors coherence
        nn_coherence = nearest_neighbors_and_coherence(kv, test_words, topn=topn)
        with open(model_out / f"{model_name}_neighbors_coherence.json", "w", encoding="utf-8") as f:
            json.dump(nn_coherence, f, ensure_ascii=False, indent=2)

        # 7) Nearest neighbors table as CSV
        rows = []
        for w, neigh in sim_matrix.items():
            for rank, (nw, score) in enumerate(neigh, start=1):
                rows.append({"word": w, "neighbor_rank": rank, "neighbor": nw, "score": float(score)})
        pd.DataFrame(rows).to_csv(model_out / f"{model_name}_test_words_neighbors.csv", index=False)

        # 8) compute neighbor coherence summary (mean over test words)
        coh_vals = [v["coherence_word_neighbors"] for v in nn_coherence.values() if v["coherence_word_neighbors"] is not None]
        pair_coh_vals = [v["coherence_neighbors_pairwise"] for v in nn_coherence.values() if v["coherence_neighbors_pairwise"] is not None]
        mean_coh = float(np.mean(coh_vals)) if coh_vals else None
        mean_pair_coh = float(np.mean(pair_coh_vals)) if pair_coh_vals else None

        # 9) similarity correlation with human judgments (if available)
        sim_corr = None
        if WORD_SIM_PATH.exists():
            try:
                df_sim = pd.read_csv(WORD_SIM_PATH)
                human_scores = []
                model_scores = []
                for _, r in df_sim.iterrows():
                    w1, w2 = str(r["word1"]), str(r["word2"])
                    if w1 in kv.key_to_index and w2 in kv.key_to_index:
                        model_scores.append(kv.similarity(w1, w2))
                        human_scores.append(float(r["human_score"]))
                if len(human_scores) >= 3:
                    from scipy.stats import spearmanr
                    corr, p = spearmanr(model_scores, human_scores)
                    sim_corr = {"spearman_r": float(corr), "pvalue": float(p), "n_pairs": len(model_scores)}
            except Exception as e:
                sim_corr = {"error": str(e)}

        # 10) assemble summary for this model
        summary = {
            "model": model_name,
            "vocab_size": len(kv.index_to_key) if hasattr(kv, "index_to_key") else len(kv.key_to_index),
            "sampled_pairwise_mean": float(np.mean(sims)) if sims.size else None,
            "sampled_pairwise_std": float(np.std(sims)) if sims.size else None,
            "synonyms_stats": syn_res,
            "antonyms_stats": ant_res,
            "morph_stats": morph_res,
            "analogies_eval": analogies_res,
            "neighbors_coherence_mean": mean_coh,
            "neighbors_pairwise_coherence_mean": mean_pair_coh,
            "similarity_correlation": sim_corr
        }
        overall_results.append(summary)

        # save summary + entire raw results for model
        with open(model_out / f"{model_name}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # small print
        print(f"Saved results for {model_name} -> {model_out}")

    # save global summary CSV
    df = pd.DataFrame(overall_results)
    df.to_csv(out_dir / "semantic_ops_summary.csv", index=False)
    with open(out_dir / "semantic_ops_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall_results, f, ensure_ascii=False, indent=2)

    print("\n✅ Stage 6 finished. Results saved in", out_dir)

# ----------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, default=str(DEFAULT_MODELS_DIR), help="Папка с моделями (data/models)")
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Папка для результатов")
    p.add_argument("--topn", type=int, default=DEFAULT_TOPN, help="сколько соседей сохранять для тестовых слов")
    p.add_argument("--sample_pairs", type=int, default=DEFAULT_SAMPLE_PAIRS, help="число сэмплов пар слов для распределения")
    p.add_argument("--test_words", type=str, default=str(DEFAULT_TEST_WORDS), help="файл со списком тестовых слов (опционально)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_experiments(models_dir=Path(args.models),
                    out_dir=Path(args.out),
                    topn=args.topn,
                    sample_pairs=args.sample_pairs,
                    test_words_path=Path(args.test_words))
