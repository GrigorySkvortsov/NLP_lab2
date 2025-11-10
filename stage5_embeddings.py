#!/usr/bin/env python3
"""
stage5_embeddings.py

Этап 5. Обучение моделей распределённых представлений и их оценка.

Что делает:
- Загружает корпус (data/docs_meta.jsonl) или corpus.jsonl при отсутствии
- Токенизирует (предполагается, что тексты уже токенизированы пробелами, как в корпусе ЛР1)
- Перебирает параметры и обучает:
    - Word2Vec (sg=0/1) via gensim
    - FastText (sg-like via gensim FastText)
    - Doc2Vec (dm=1/0)
    - (опционально GloVe, если доступна библиотека glove)
- Оценивает модели: время обучения, размер файла, память (wv vectors), coverage, analogy (если data/analogies.txt),
  similarity correlation (если data/word_similarity.csv), nearest neighbors (test words list), а для doc-эмбеддингов — ARI + классификация (если есть category label).
- Сохраняет модели в data/models/ и метрики в outputs/embeddings_metrics.csv,
  а также файлы соседей в outputs/nearest_neighbors/.

Зависимости:
    pip install gensim scikit-learn pandas numpy joblib
    (optional) pip install glove_python_binary umap-learn

Запуск:
    python stage5_embeddings.py
"""

import os
import time
import json
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# gensim
from gensim.models import Word2Vec, FastText, Doc2Vec
from gensim.models.doc2vec import TaggedDocument

# sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr

# filesystem
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
MODELS_DIR = DATA_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "nearest_neighbors").mkdir(exist_ok=True)

# parameters grid (can be adjusted)
VECTOR_SIZES = [100, 200, 300]
WINDOWS = [5, 8, 10]
MIN_COUNTS = [5, 10]
WORD2VEC_SGS = [0, 1]  # 0=CBOW, 1=Skip-gram
FASTTEXT_MODELS = ["cbow", "skipgram"]  # gensim uses sg param; we'll map
DOC2VEC_DMS = [1, 0]  # 1=PV-DM, 0=PV-DBOW

# evaluation resources (optional)
ANALOGIES_PATH = DATA_DIR / "analogies.txt"         # word2vec-style analogy file
SIMILARITY_PAIRS = DATA_DIR / "word_similarity.csv" # csv with columns word1,word2,human_score
MORPH_PAIRS = DATA_DIR / "morph_pairs.csv"          # optional morphological pairs for robustness
TEST_WORDS = DATA_DIR / "test_words.txt"            # optional list of words to save nearest neighbors

# helper: load corpus (token lists) and docs_meta
def load_texts_and_meta():
    docs_meta_path = DATA_DIR / "docs_meta.jsonl"
    texts = []
    docs_meta = None
    if docs_meta_path.exists():
        docs_meta = []
        with open(docs_meta_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                docs_meta.append(obj)
                texts.append(obj.get("text", ""))
    else:
        # fallback to corpus.jsonl
        corpus_path = Path("input/corpus.jsonl")
        with open(corpus_path, "r", encoding="utf-8") as f:
            docs_meta = []
            for line in f:
                obj = json.loads(line)
                docs_meta.append(obj)
                texts.append(obj.get("text", ""))
    # assume tokens separated by whitespace (as in your sample)
    tokenized = [t.split() if isinstance(t, str) else [] for t in texts]
    return tokenized, docs_meta

# coverage: fraction of unique corpus tokens found in model
def compute_coverage(model_wv, corpus_token_lists):
    corpus_vocab = set()
    for tokens in corpus_token_lists:
        corpus_vocab.update(tokens)
    total = len(corpus_vocab)
    present = sum(1 for w in corpus_vocab if w in model_wv.key_to_index)
    return float(present) / total if total > 0 else None, total, present

# analogy evaluation wrapper (uses gensim evaluate_word_analogies if file present)
def evaluate_analogies_if_exists(model, path: Path):
    if not path.exists():
        return None
    try:
        # gensim KeyedVectors has evaluate_word_analogies; models have .wv
        res = model.wv.evaluate_word_analogies(str(path))
        # res is (section_results, overall_correct, overall_total)
        # we'll compute overall accuracy if possible
        overall = res[0]
        # gensim >=4 returns a dict for section_results and tuple (correct, incorrect) maybe in res[1]
        # to be robust:
        try:
            overall_scores = res[1]  # sometimes (correct, incorrect)
            correct = overall_scores[0]
            incorrect = overall_scores[1]
            accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else None
        except Exception:
            # fallback: inspect structure
            accuracy = None
        return {"raw": res, "accuracy": accuracy}
    except Exception as e:
        return {"error": str(e)}

# similarity correlation
def similarity_correlation_if_exists(model_wv, path: Path):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        # expects columns: word1,word2,human_score
        sims = []
        human = []
        for _, row in df.iterrows():
            w1 = row["word1"]
            w2 = row["word2"]
            if w1 in model_wv.key_to_index and w2 in model_wv.key_to_index:
                sims.append(model_wv.similarity(w1, w2))
                human.append(float(row["human_score"]))
        if len(sims) < 3:
            return None
        corr, p = spearmanr(sims, human)
        return {"spearman_r": float(corr), "pvalue": float(p), "n_pairs": len(sims)}
    except Exception as e:
        return {"error": str(e)}

# nearest neighbors for test words
def save_nearest_neighbors(model_wv, words_path: Path, out_path: Path, topn=10):
    out = {}
    if not words_path.exists():
        return None
    with open(words_path, "r", encoding="utf-8") as f:
        words = [w.strip() for w in f if w.strip()]
    for w in words:
        if w in model_wv.key_to_index:
            try:
                neigh = model_wv.most_common(w, topn=topn) if hasattr(model_wv, "most_common") else model_wv.most_similar(w, topn=topn)
                # gensim KV: most_similar returns list of (word, score)
                if isinstance(neigh, dict):  # if most_common returns dict in some APIs
                    neighbors = list(neigh.items())[:topn]
                else:
                    neighbors = neigh
                out[w] = neighbors
            except Exception as e:
                out[w] = {"error": str(e)}
        else:
            out[w] = None
    # save JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out

# train helpers
def train_word2vec(sentences, vector_size, window, min_count, sg, epochs=5, workers=4):
    start = time.time()
    model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count, sg=sg, workers=workers)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)
    elapsed = time.time() - start
    return model, elapsed

def train_fasttext(sentences, vector_size, window, min_count, model_type, epochs=5, workers=4):
    # gensim FastText: sg param same as Word2Vec (sg=1 -> skipgram). 'model_type' == 'skipgram'/'cbow'
    sg = 1 if model_type == "skipgram" else 0
    start = time.time()
    model = FastText(vector_size=vector_size, window=window, min_count=min_count, sg=sg, workers=workers)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)
    elapsed = time.time() - start
    return model, elapsed

def train_doc2vec(tagged_docs, vector_size, window, min_count, dm, epochs=20, workers=4):
    start = time.time()
    model = Doc2Vec(vector_size=vector_size, window=window, min_count=min_count, dm=dm, workers=workers)
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=epochs)
    elapsed = time.time() - start
    return model, elapsed

# safe save size
def model_disk_size(path: Path):
    if path.exists():
        return path.stat().st_size
    return None

def approx_wv_memory(model):
    try:
        return int(model.wv.vectors.nbytes)
    except Exception:
        try:
            return int(model.wv.vectors_cpu.nbytes)
        except Exception:
            return None

def run():
    # load texts
    print("Loading corpus...")
    tokenized_texts, docs_meta = load_texts_and_meta()
    n_docs = len(tokenized_texts)
    print(f"Loaded {n_docs} documents (tokenized).")

    # prepare tagged docs for Doc2Vec
    tagged = [TaggedDocument(words=t, tags=[str(i)]) for i, t in enumerate(tokenized_texts)]

    metrics = []  # will collect dicts per trained model

    # optional evaluation resources
    analogies_exists = ANALOGIES_PATH.exists()
    sim_pairs_exists = SIMILARITY_PAIRS.exists()
    test_words_exists = TEST_WORDS.exists()

    # TRAIN WORD-LEVEL MODELS
    for vector_size in VECTOR_SIZES:
        for window in WINDOWS:
            for min_count in MIN_COUNTS:
                # Word2Vec both sg values
                for sg in WORD2VEC_SGS:
                    name = f"word2vec_sg{sg}_vs{vector_size}_w{window}_mc{min_count}"
                    print(f"\nTraining {name} ...")
                    try:
                        model, t_elapsed = train_word2vec(tokenized_texts, vector_size, window, min_count, sg, epochs=10)
                    except Exception as e:
                        print(f"  Error training {name}: {e}")
                        continue
                    # save
                    model_path = MODELS_DIR / f"{name}.model"
                    model.save(str(model_path))
                    disk = model_disk_size(model_path)
                    mem = approx_wv_memory(model)
                    coverage, total_vocab, present = compute_coverage(model.wv, tokenized_texts)

                    # analogy eval if available
                    analogy_res = evaluate_analogies_if_exists(model, ANALOGIES_PATH) if analogies_exists else None
                    sim_corr = similarity_correlation_if_exists(model.wv, SIMILARITY_PAIRS) if sim_pairs_exists else None
                    # nearest neighbors
                    if test_words_exists:
                        nn = save_nearest_neighbors(model.wv, TEST_WORDS, OUTPUT_DIR / "nearest_neighbors" / f"{name}_neighbors.json", topn=10)
                    else:
                        nn = None

                    metrics.append({
                        "model_name": name,
                        "family": "word2vec",
                        "vector_size": vector_size,
                        "window": window,
                        "min_count": min_count,
                        "arch": "sg" if sg==1 else "cbow",
                        "train_time_sec": t_elapsed,
                        "disk_bytes": disk,
                        "wv_mem_bytes": mem,
                        "coverage": coverage,
                        "corpus_vocab_total": total_vocab,
                        "corpus_vocab_present": present,
                        "analogy": analogy_res,
                        "similarity_corr": sim_corr,
                        "nearest_neighbors_file": str(OUTPUT_DIR / "nearest_neighbors" / f"{name}_neighbors.json") if nn else None
                    })

                # FastText (gensim)
                for ftm in FASTTEXT_MODELS:
                    name = f"fasttext_{ftm}_vs{vector_size}_w{window}_mc{min_count}"
                    print(f"\nTraining {name} ...")
                    try:
                        model, t_elapsed = train_fasttext(tokenized_texts, vector_size, window, min_count, ftm, epochs=5)
                    except Exception as e:
                        print(f"  Error training {name}: {e}")
                        continue
                    # save
                    model_path = MODELS_DIR / f"{name}.model"
                    model.save(str(model_path))
                    disk = model_disk_size(model_path)
                    mem = approx_wv_memory(model)
                    coverage, total_vocab, present = compute_coverage(model.wv, tokenized_texts)
                    analogy_res = evaluate_analogies_if_exists(model, ANALOGIES_PATH) if analogies_exists else None
                    sim_corr = similarity_correlation_if_exists(model.wv, SIMILARITY_PAIRS) if sim_pairs_exists else None
                    if test_words_exists:
                        nn = save_nearest_neighbors(model.wv, TEST_WORDS, OUTPUT_DIR / "nearest_neighbors" / f"{name}_neighbors.json", topn=10)
                    else:
                        nn = None

                    metrics.append({
                        "model_name": name,
                        "family": "fasttext",
                        "vector_size": vector_size,
                        "window": window,
                        "min_count": min_count,
                        "arch": ftm,
                        "train_time_sec": t_elapsed,
                        "disk_bytes": disk,
                        "wv_mem_bytes": mem,
                        "coverage": coverage,
                        "corpus_vocab_total": total_vocab,
                        "corpus_vocab_present": present,
                        "analogy": analogy_res,
                        "similarity_corr": sim_corr,
                        "nearest_neighbors_file": str(OUTPUT_DIR / "nearest_neighbors" / f"{name}_neighbors.json") if nn else None
                    })

    # DOC-LEVEL MODELS (Doc2Vec)
    for vector_size in VECTOR_SIZES:
        for window in WINDOWS:
            for min_count in MIN_COUNTS:
                for dm in DOC2VEC_DMS:
                    name = f"doc2vec_dm{dm}_vs{vector_size}_w{window}_mc{min_count}"
                    print(f"\nTraining {name} ...")
                    try:
                        model, t_elapsed = train_doc2vec(tagged, vector_size, window, min_count, dm, epochs=20)
                    except Exception as e:
                        print(f"  Error training {name}: {e}")
                        continue
                    model_path = MODELS_DIR / f"{name}.model"
                    model.save(str(model_path))
                    disk = model_disk_size(model_path)
                    # doc2vec memory: approximate by docvecs + wv
                    try:
                        mem = int(model.wv.vectors.nbytes + model.docvecs.vectors_docs.nbytes)
                    except Exception:
                        mem = approx_wv_memory(model)
                    # prepare document vectors
                    doc_vectors = np.array([model.dv[str(i)] for i in range(len(tokenized_texts))])
                    # clustering ARI if categories exist
                    ari = None
                    clf_acc = None
                    if docs_meta is not None and all("category" in d for d in docs_meta):
                        y = [d["category"] for d in docs_meta]
                        uniq = sorted(set(y))
                        y_idx = np.array([uniq.index(xx) for xx in y])
                        n_clusters = min(len(uniq), 20)
                        try:
                            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                            y_pred = km.fit_predict(doc_vectors)
                            ari = float(adjusted_rand_score(y_idx, y_pred))
                        except Exception as e:
                            ari = None
                        # classification accuracy via logistic regression cross-val
                        try:
                            clf = LogisticRegression(max_iter=1000)
                            scores = cross_val_score(clf, doc_vectors, y_idx, cv=5, scoring="accuracy")
                            clf_acc = float(np.mean(scores))
                        except Exception:
                            clf_acc = None

                    metrics.append({
                        "model_name": name,
                        "family": "doc2vec",
                        "vector_size": vector_size,
                        "window": window,
                        "min_count": min_count,
                        "arch": f"dm{dm}",
                        "train_time_sec": t_elapsed,
                        "disk_bytes": disk,
                        "wv_mem_bytes": mem,
                        "ari": ari,
                        "doc_classification_accuracy_cv5": clf_acc
                    })

    # Save metrics to CSV/JSON
    metrics_out = OUTPUT_DIR / "embeddings_metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # flatten simple table for CSV summary (only scalar fields)
    rows = []
    for m in metrics:
        row = {k: v for k, v in m.items() if isinstance(v, (int, float, str, type(None)))}
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "embeddings_metrics.csv", index=False)

    print("\n✅ Stage 5 finished.")
    print(f"Models saved to: {MODELS_DIR}")
    print(f"Metrics saved to: {OUTPUT_DIR / 'embeddings_metrics.csv'} and .json")

if __name__ == "__main__":
    run()
