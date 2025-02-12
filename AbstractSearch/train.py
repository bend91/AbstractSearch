from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
import os
from .utils import reciprocal_rank_fusion


# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
NLTK_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")
BM25_MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "bm25_model.pkl")
INDEX_MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "abstracts.index")
EMBED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "embed_model")
ABSTRACT_PATH = os.path.join(os.path.dirname(__file__), "data", "Car_abstracts.csv")
SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path = [NLTK_DIR]



try:
    nltk.download("punkt_tab", download_dir=NLTK_DIR, quiet=True)
    nltk.download("stopwords", download_dir=NLTK_DIR, quiet=True)
except LookupError:
    raise LookupError("NLTK data download failed, Please check your internet connection")

STOPWORDS = set(stopwords.words("english"))

def bm25_training(df):
    abstract_bodies = df["Title"].values + df["Body"].values

    abstract_bodies = [preprocess_text(abstract, STOPWORDS) for abstract in abstract_bodies]
    print("training BM25Okapi")
    bm25 = BM25Okapi(abstract_bodies)
    with open(BM25_MODEL_PATH, "wb") as f:
        pickle.dump(bm25, f)


def faiss_training(df):
    abstract_bodies = df["Title"].values + df["Body"].values
    print("encoding abstracts")
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    try:
        embed_model.save(EMBED_MODEL_PATH)
    except FileExistsError:
        embed_model = SentenceTransformer(EMBED_MODEL_PATH)
    embeddings = embed_model.encode(abstract_bodies)
    d = embeddings.shape[1]
    print("indexing embeddings")
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings, dtype=np.float32))
    print("writing indexes")
    faiss.write_index(index, INDEX_MODEL_PATH)


def preprocess_text(text, stopwords=STOPWORDS):
    return [word for word in word_tokenize(text.lower()) if word not in stopwords]


def training(df):
    faiss_training(df)
    bm25_training(df)
