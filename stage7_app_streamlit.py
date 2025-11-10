# stage7_app_streamlit.py
"""
Streamlit приложение для Этапа 7:
Интерактивный анализ векторных пространств (векторная арифметика, сходство, оси и визуализации).

Требования:
    pip install streamlit gensim scikit-learn umap-learn plotly pandas numpy
(umap-learn опционален; если не установлен, используется TSNE)

Запуск:
    streamlit run stage7_app_streamlit.py
"""

import streamlit as st
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import time
import os
from typing import List, Tuple, Dict, Any

# gensim
from gensim.models import Word2Vec, FastText, Doc2Vec, KeyedVectors

# sklearn / umap / tsne
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
try:
    import umap
    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

import plotly.express as px
import plotly.graph_objects as go

import re
from sklearn.preprocessing import normalize as sk_normalize

# нормализация вектора (L2)
def normalize_vec(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec
    return vec / norm

# безопасно получить частоту слова (gensim >=4: get_vecattr)
def get_word_count(kv, word: str):
    try:
        return int(kv.get_vecattr(word, "count"))
    except Exception:
        try:
            # gensim <4
            return int(kv.vocab[word].count)
        except Exception:
            return None

# фильтр: оставлять только кириллические токены (опционально)
CYRILLIC_RE = re.compile(r"^[\u0400-\u04FFёЁ\-]+$")

def is_cyrillic(word: str) -> bool:
    return bool(CYRILLIC_RE.match(word))

# Параметры фильтрации
# only_cyrillic = True
# min_count = 5
# topn = 10  # количество ближайших слов


# ------------------- Конфигурация путей -------------------
DATA_DIR = Path("data")
MODELS_DIR = DATA_DIR / "models"
VECTOR_META_PATH = DATA_DIR / "vectorization_meta.json"
VECTORIZERS_DIR = DATA_DIR  # vectorizer pickles are saved here as {config}_vectorizer.pkl
DOCS_META = DATA_DIR / "docs_meta.jsonl"
ANALOGIES_PATH = DATA_DIR / "analogies.txt"
# ---------------------------------------------------------

st.set_page_config(page_title="NLP: Vector Space Explorer", layout="wide")

# ------------------ Утилиты для загрузки ------------------
@st.cache_resource(show_spinner=False)
def list_models() -> List[str]:
    if not MODELS_DIR.exists():
        return []
    files = sorted([p.name for p in MODELS_DIR.glob("*.model")] + [p.name for p in MODELS_DIR.glob("*.kv")])
    return files

@st.cache_resource(show_spinner=False)
def load_gensim_model(path: Path):
    # Попробуем загрузить модель в порядке совместимости
    try:
        return Word2Vec.load(str(path))
    except Exception:
        try:
            return FastText.load(str(path))
        except Exception:
            try:
                return Doc2Vec.load(str(path))
            except Exception:
                try:
                    return KeyedVectors.load(str(path))
                except Exception:
                    try:
                        return KeyedVectors.load_word2vec_format(str(path), binary=True)
                    except Exception as e:
                        st.warning(f"Не удалось загрузить модель {path.name}: {e}")
                        return None

@st.cache_resource(show_spinner=False)
def load_vectorizer(config_name: str):
    p = VECTORIZERS_DIR / f"{config_name}_vectorizer.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def load_docs_meta():
    if not DOCS_META.exists():
        return None
    docs = []
    with open(DOCS_META, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs

@st.cache_data(show_spinner=False)
def load_vectorization_meta():
    if not VECTOR_META_PATH.exists():
        return []
    with open(VECTOR_META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# ------------------ Основные операции над векторами ------------------
def parse_expression(expr: str) -> List[Tuple[str, int]]:
    """
    Простейший парсер выражений вида: 'король - мужчина + женщина - ...'
    Возвращает список (token, sign) в порядке выполнения.
    sign = +1 или -1
    """
    # удаляем лишние пробелы, заменяем минус/плюс через пробелы
    toks = expr.replace('-', ' - ').replace('+', ' + ').split()
    out = []
    sign = +1
    for t in toks:
        if t == '+':
            sign = +1
        elif t == '-':
            sign = -1
        else:
            out.append((t, sign))
            sign = +1
    return out

def compute_expression_vector(kv, expr_tokens: List[Tuple[str,int]]):
    # """
    # Возвращает список промежуточных векторов и итоговый вектор.
    # Если слово отсутствует в kv, оно пропускается и в результате указывается ошибка.
    # """
    # vectors = []
    # words_present = []
    # warnings = []
    # acc = np.zeros(kv.vector_size, dtype=np.float32)
    # for w, s in expr_tokens:
    #     if w in kv.key_to_index:
    #         vec = kv[w] * s
    #         acc = acc + vec
    #         words_present.append((w, s))
    #         vectors.append((w, s, acc.copy()))
    #     else:
    #         warnings.append(f"Слово '{w}' не найдено в модели (пропущено).")
    # return vectors, acc, warnings

    """
    Возвращает:
      - vectors: список (word, sign, intermediate_acc_vector)
      - acc_vec: итоговый (не нормализованный) вектор (но все части нормализованы при добавлении)
      - warnings: список сообщений о пропусках
    """
    vectors = []
    warnings = []
    acc = np.zeros(kv.vector_size, dtype=np.float32)
    for w, s in expr_tokens:
        if w in kv.key_to_index:
            vec = kv[w]
            vec_n = normalize_vec(vec)
            acc = acc + vec_n * s
            vectors.append((w, s, acc.copy()))
        else:
            warnings.append(f"Слово '{w}' не найдено в модели (пропущено).")
    return vectors, acc, warnings

def topn_most_similar(kv, vec, topn=10, exclude=set(), min_count=5, only_cyrillic=True):
    # """
    # Если kv поддерживает most_similar с вектором, можно использовать fallback:
    # вычисляем косинус вручную относительно словаря.
    # """
    # try:
    #     # Try using KeyedVectors.similar_by_vector if available
    #     if hasattr(kv, "similar_by_vector"):
    #         return kv.similar_by_vector(vec, topn=topn)
    # except Exception:
    #     pass
    # # fallback manual
    # words = kv.index_to_key if hasattr(kv, "index_to_key") else list(kv.key_to_index.keys())
    # sims = []
    # vnorm = np.linalg.norm(vec) + 1e-12
    # for w in words:
    #     if w in exclude:
    #         continue
    #     try:
    #         vw = kv[w]
    #         sims.append((w, float(np.dot(vec, vw) / (vnorm * (np.linalg.norm(vw) + 1e-12)))))
    #     except Exception:
    #         continue
    # sims.sort(key=lambda x: -x[1])
    # return sims[:topn]

    exclude = exclude or set()
    # нормализуем входной вектор
    vec_n = normalize_vec(vec)
    try:
        # gensim method (нормализует внутри)
        candidates = kv.similar_by_vector(vec_n, topn=topn + len(exclude))
    except Exception:
        # fallback: ручной перебор
        words = kv.index_to_key
        sims = []
        for w in words:
            if w in exclude:
                continue
            try:
                sims.append((w, float(np.dot(vec_n, normalize_vec(kv[w])))))
            except Exception:
                continue
        sims.sort(key=lambda x: -x[1])
        candidates = sims[:topn + len(exclude)]

    out = []
    for w, score in candidates:
        if w in exclude:
            continue
        if only_cyrillic and not is_cyrillic(w):
            continue
        cnt = get_word_count(kv, w)
        if min_count is not None and cnt is not None and cnt < min_count:
            continue
        out.append((w, float(score)))
        if len(out) >= topn:
            break
    return out

# ------------------ UI: Sidebar ------------------
st.sidebar.title("Настройки")
models = list_models()
model_choice = st.sidebar.selectbox("Выбери модель (word embeddings / doc2vec) из data/models", options=["(none)"] + models)
use_vectorizers = st.sidebar.checkbox("Использовать классические векторизаторы (TF-IDF/BoW) из data/", value=True)
vector_meta = load_vectorization_meta()
vector_configs = [m["config"] for m in vector_meta] if vector_meta else []
vector_choice = st.sidebar.selectbox("Выбери конфигурацию векторизатора", options=["(none)"] + vector_configs)

# options
topn = st.sidebar.number_input("top-N ближайших соседей", min_value=1, max_value=50, value=10)
umap_n_neighbors = st.sidebar.slider("UMAP: n_neighbors (если доступен)", 5, 200, 15)
umap_min_dist = st.sidebar.slider("UMAP: min_dist", 0.0, 1.0, 0.1, 0.05)

docs_meta = load_docs_meta()

# ------------------ Main Layout ------------------
st.title("Vector Space Explorer — интерактивный анализ векторных представлений")
st.markdown("Интерфейс для анализа векторных пространств: арифметика, сходство, оси, визуализации. \
             Используются модели в `data/models/` и векторизаторы в `data/`.")

col1, col2 = st.columns([2,1])

with col2:
    st.header("Информация")
    st.write("Загружено моделей:", len(models))
    st.write("Векторизаторов конфигов:", len(vector_configs))
    if docs_meta:
        st.write("Документов:", len(docs_meta))
        # show sample categories if present
        if "category" in docs_meta[0]:
            cats = sorted({d.get("category") for d in docs_meta})
            st.write("Присутствуют категории:", len(cats))

# ------------------ Tabs ------------------
tabs = st.tabs(["Векторная арифметика", "Калькулятор сходства", "Семантические оси / bias", "Визуализация", "Генерация отчёта"])

# ---------- TAB 1: Vector arithmetic ----------
with tabs[0]:
    st.header("Интерактивная векторная арифметика")
    st.write("Формат ввода: `король - мужчина + женщина` (простая арифметика).")
    expr = st.text_input("Введи выражение", value="король - мужчина + женщина")
    btn_run = st.button("Вычислить")
    if btn_run:
        if model_choice == "(none)":
            st.warning("Выберите модель в панели слева.")
        else:
            model_path = MODELS_DIR / model_choice
            model = load_gensim_model(model_path)
            kv = model.wv if hasattr(model, "wv") else (model if isinstance(model, KeyedVectors) else None)
            if kv is None:
                st.error("Не удалось получить KeyedVectors из выбранной модели.")
            else:
                # expr_tokens = parse_expression(expr)
                # vectors, acc_vec, warnings = compute_expression_vector(kv, expr_tokens)
                # st.subheader("Промежуточные шаги")
                # for i, (w, s, vec) in enumerate(vectors, start=1):
                #     st.write(f"Шаг {i}: {'+' if s>0 else '-'} {w}")
                #     # show top neighbors for this intermediate vector
                #     neigh = topn_most_similar(kv, vec, topn=topn, exclude={w})
                #     df = pd.DataFrame(neigh, columns=["word","score"])
                #     st.dataframe(df)
                # if warnings:
                #     for w in warnings:
                #         st.warning(w)
                # st.subheader("Результат (аккумуляция всего выражения)")
                # st.write(f"Вектор L2-норма: {np.linalg.norm(acc_vec):.4f}")
                # final_neigh = topn_most_similar(kv, acc_vec, topn=topn)
                # st.markdown("**Топ ближайших слов (результат):**")
                # st.dataframe(pd.DataFrame(final_neigh, columns=["word","score"]))

                expr_tokens = parse_expression(expr)
                vectors, acc_vec, warnings = compute_expression_vector(kv, expr_tokens)

                st.subheader("Промежуточные шаги (нормализованные векторы)")
                for i, (w, s, vec) in enumerate(vectors, start=1):
                    st.write(f"Шаг {i}: {'+' if s>0 else '-'} {w}")
                    neigh = topn_most_similar(kv, vec, topn=topn, exclude=set(), min_count=5, only_cyrillic=True)
                    st.dataframe(pd.DataFrame(neigh, columns=["word","score"]))

                if warnings:
                    for wmsg in warnings:
                        st.warning(wmsg)

                st.subheader("Результат (используем gensim most_similar при возможности)")
                # соберём positive/negative списки из выражения (исключая пропущенные)
                positive = [w for w,s in expr_tokens if s>0 and w in kv.key_to_index]
                negative = [w for w,s in expr_tokens if s<0 and w in kv.key_to_index]

                try:
                    # gensim сам делает нормализацию, чаще даёт лучшее семантическое соответствие
                    if positive or negative:
                        final = kv.most_similar(positive=positive, negative=negative, topn=topn)
                    else:
                        final = topn_most_similar(kv, acc_vec, topn=topn)
                except Exception:
                    final = topn_most_similar(kv, acc_vec, topn=topn)

                # дополнительно фильтруем result: кириллица + min_count
                final_filtered = []
                for w, sc in final:
                    if not is_cyrillic(w):
                        continue
                    cnt = get_word_count(kv, w)
                    if cnt is not None and cnt < 5:
                        continue
                    final_filtered.append((w, float(sc)))
                    if len(final_filtered) >= topn:
                        break

                st.dataframe(pd.DataFrame(final_filtered, columns=["word","score"]))

# ---------- TAB 2: Similarity calculator ----------
with tabs[1]:
    st.header("Калькулятор косинусного сходства")
    st.write("Поддерживает word embeddings (рекомендуется). Для векторизаторов — вычисляем сходство между документами (по индексу).")
    colA, colB = st.columns(2)
    with colA:
        word_a = st.text_input("Слово A", value="франция")
    with colB:
        word_b = st.text_input("Слово B", value="великобритания")
    method = st.radio("Источник векторов", options=["Embeddings (selected model)", "Vectorizer (document vectors)"])
    if st.button("Посчитать сходство"):
        if method.startswith("Embeddings"):
            if model_choice == "(none)":
                st.warning("Выберите модель в сайдбаре.")
            else:
                model = load_gensim_model(MODELS_DIR / model_choice)
                kv = model.wv if hasattr(model, "wv") else model
                if word_a in kv.key_to_index and word_b in kv.key_to_index:
                    sim = kv.similarity(word_a, word_b)
                    st.success(f"Косинусное сходство (embeddings) между `{word_a}` и `{word_b}` = {sim:.4f}")
                    # show neighbors of each
                    st.write("Соседи для A:")
                    # st.dataframe(pd.DataFrame(kv.most_similar(word_a, topn=topn), columns=["word","score"]))

                    raw = kv.most_similar(word_a, topn=topn*3)  # возьмём немного больше
                    neighbors = []
                    for w, sc in raw:
                        if not is_cyrillic(w):
                            continue
                        cnt = get_word_count(kv, w)
                        if cnt is not None and cnt < 5:
                            continue
                        neighbors.append((w, float(sc)))
                        if len(neighbors) >= topn:
                            break
                    st.dataframe(pd.DataFrame(neighbors, columns=["word","score"]))

                    st.write("Соседи для B:")
                    # st.dataframe(pd.DataFrame(kv.most_similar(word_b, topn=topn), columns=["word","score"]))

                    raw = kv.most_similar(word_b, topn=topn*3)  # возьмём немного больше
                    neighbors = []
                    for w, sc in raw:
                        if not is_cyrillic(w):
                            continue
                        cnt = get_word_count(kv, w)
                        if cnt is not None and cnt < 5:
                            continue
                        neighbors.append((w, float(sc)))
                        if len(neighbors) >= topn:
                            break
                    st.dataframe(pd.DataFrame(neighbors, columns=["word","score"]))

                else:
                    st.error("Одно из слов отсутствует в модели.")
        else:
            # document similarity via vectorizer: let user choose two doc indices
            if vector_choice == "(none)":
                st.warning("Выберите конфигурацию векторизатора в сайдбаре.")
            else:
                vec = load_vectorizer(vector_choice)
                if vec is None:
                    st.error("Не удалось загрузить vectorizer.")
                else:
                    texts = [d.get("text","") for d in (docs_meta if docs_meta else [])]
                    if not texts:
                        st.error("Нет документов (docs_meta отсутствует).")
                    else:
                        # pick indices
                        idx_a = st.number_input("Doc index A (0-based)", min_value=0, max_value=len(texts)-1, value=0)
                        idx_b = st.number_input("Doc index B (0-based)", min_value=0, max_value=len(texts)-1, value=1)
                        X = vec.transform(texts)
                        
                        # va = X[idx_a].toarray().ravel()
                        # vb = X[idx_b].toarray().ravel()
                        # sim = float(np.dot(va,vb) / ((np.linalg.norm(va)+1e-12)*(np.linalg.norm(vb)+1e-12)))

                        from sklearn.metrics.pairwise import cosine_similarity
                        va = X[idx_a]
                        vb = X[idx_b]
                        sim = float(cosine_similarity(va, vb)[0,0])


                        st.success(f"Косинус (документы) = {sim:.4f}")
                        st.write("Отображаем превью документов:")
                        st.write("Doc A (title):", docs_meta[idx_a].get("title","(нет)"))
                        st.write("Doc B (title):", docs_meta[idx_b].get("title","(нет)"))

# ---------- TAB 3: Semantic axes / Bias ----------
with tabs[2]:
    st.header("Семантические оси и оценка смещения (bias)")
    st.write("Определи ось через пару слов (A - B). Проекция других слов на эту ось покажет оценку по этой семантической шкале.")
    axis_word1 = st.text_input("Ось: положительная сторона (например 'она' / 'женщина')", value="женщина")
    axis_word2 = st.text_input("Ось: отрицательная сторона (например 'он' / 'мужчина')", value="мужчина")
    target_words_input = st.text_area("Слова для проекции (через пробел или запятую). Оставь пустым для топ-500 слов.", value="сын дочь отец мать")
    if st.button("Вычислить проекции"):
        if model_choice == "(none)":
            st.warning("Выберите модель.")
        else:
            model = load_gensim_model(MODELS_DIR / model_choice)
            kv = model.wv if hasattr(model, "wv") else model
            if axis_word1 not in kv.key_to_index or axis_word2 not in kv.key_to_index:
                st.error("Одна из опорных слов отсутствует в модели.")
            else:
                axis = kv[axis_word1] - kv[axis_word2]
                if target_words_input.strip():
                    tokens = [w.strip() for w in target_words_input.replace(",", " ").split()]
                else:
                    tokens = kv.index_to_key[:500]
                proj = {}
                axis_norm = axis / (np.linalg.norm(axis)+1e-12)
                for w in tokens:
                    if w in kv.key_to_index:
                        proj[w] = float(np.dot(kv[w], axis_norm))
                # sort and display
                dfp = pd.DataFrame(sorted(proj.items(), key=lambda x: -x[1]), columns=["word","projection"])
                st.write("Top (positive side) / Bottom (negative side):")
                st.dataframe(dfp.head(30))
                st.dataframe(dfp.tail(30))
                # bar chart of top N
                nvis = min(40, len(dfp))
                fig = px.bar(dfp.head(nvis), x="word", y="projection", title="Top projections (positive side)")
                st.plotly_chart(fig, use_container_width=True)
                fig2 = px.bar(dfp.tail(nvis).sort_values("projection"), x="word", y="projection", title="Top projections (negative side)")
                st.plotly_chart(fig2, use_container_width=True)

# ---------- TAB 4: Visualization ----------
with tabs[3]:
    st.header("2D/3D визуализация векторного пространства")
    st.write("Выберите источник: эмбеддинги (модель) или SVD от TF-IDF (vectorizer).")
    vis_source = st.selectbox("Источник для визуализации", options=["(none)", "Embeddings (model)", "SVD(TF-IDF vectorizer)"])
    if vis_source == "Embeddings (model)":
        if model_choice == "(none)":
            st.warning("Выбери модель в сайдбаре.")
        else:
            model = load_gensim_model(MODELS_DIR / model_choice)
            kv = model.wv if hasattr(model, "wv") else model
            n_points = st.slider("Число слов для отрисовки (топ по частоте)", 100, 5000, 1000, step=100)
            use_umap = st.checkbox("Использовать UMAP (если доступен)", value=UMAP_AVAILABLE)
            sample_words = kv.index_to_key[:n_points]
            vectors = np.array([kv[w] for w in sample_words])
            if use_umap and UMAP_AVAILABLE:
                reducer = umap.UMAP(n_components=2, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, random_state=42)
                emb2d = reducer.fit_transform(vectors)
            else:
                tsne = TSNE(n_components=2, perplexity=30, init="random", random_state=42)
                emb2d = tsne.fit_transform(vectors)
            df_vis = pd.DataFrame({"word": sample_words, "x": emb2d[:,0], "y": emb2d[:,1]})
            fig = px.scatter(df_vis, x="x", y="y", hover_name="word", height=700)
            st.plotly_chart(fig, use_container_width=True)

    elif vis_source == "SVD(TF-IDF vectorizer)":
        if vector_choice == "(none)":
            st.warning("Выбери vectorizer config в сайдбаре")
        else:
            vec = load_vectorizer(vector_choice)
            if vec is None:
                st.error("Не удалось загрузить vectorizer.")
            else:
                # load texts
                if not docs_meta:
                    st.error("Нет docs_meta; перезапусти stage2 чтобы сохранить docs_meta.jsonl")
                else:
                    texts = [d.get("text","") for d in docs_meta]
                    X = vec.transform(texts)
                    n_components = st.slider("Число компонент SVD", 2, 300, 50)
                    svd = TruncatedSVD(n_components=n_components, random_state=42)
                    Xred = svd.fit_transform(X)
                    use_umap = st.checkbox("UMAP по результатам SVD (если доступен)", value=False)
                    if use_umap and UMAP_AVAILABLE:
                        reducer = umap.UMAP(n_components=2, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, random_state=42)
                        emb2d = reducer.fit_transform(Xred)
                    else:
                        tsne = TSNE(n_components=2, perplexity=30, init="random", random_state=42)
                        emb2d = tsne.fit_transform(Xred)
                    df_vis = pd.DataFrame({"doc_index": list(range(len(texts))), "x": emb2d[:,0], "y": emb2d[:,1]})
                    # color by category if available
                    if "category" in docs_meta[0]:
                        df_vis["category"] = [d.get("category") for d in docs_meta]
                    fig = px.scatter(df_vis, x="x", y="y", hover_name="doc_index", color=df_vis.get("category", None), height=700)
                    st.plotly_chart(fig, use_container_width=True)
                    st.write("Можно кликнуть по точке на графике, затем посмотреть текст документа (еще можно добавить callback).")

# ---------- TAB 5: Report / Export ----------
with tabs[4]:
    st.header("Генерация отчёта и экспорт результатов")
    st.write("Вы можете скачать результаты последних операций (если они были сохранены в outputs/).")
    out_dir = Path("outputs")
    files = []
    if out_dir.exists():
        for p in out_dir.rglob("*"):
            if p.is_file() and p.suffix in [".csv", ".json", ".png"]:
                files.append(p)
    file_choice = st.selectbox("Выбери файл для скачивания (outputs/)", options=["(none)"] + [str(p.relative_to(".")) for p in files])
    if st.button("Скачать выбранный файл"):
        if file_choice == "(none)":
            st.info("Выбери файл.")
        else:
            with open(file_choice, "rb") as f:
                data = f.read()
            st.download_button(label="Скачать", data=data, file_name=Path(file_choice).name)

st.sidebar.markdown("---")
st.sidebar.markdown("Разработано для лабораторной работы. Код открыт для модификаций.")
