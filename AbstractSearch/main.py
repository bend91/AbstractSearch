# from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
import os
from multiprocessing import Pool
from .utils import reciprocal_rank_fusion
from .train import ABSTRACT_PATH, INDEX_MODEL_PATH, BM25_MODEL_PATH, EMBED_MODEL_PATH, training



class TestModel:
    def __init__(self, *args, **kwargs):
        self.model_path = EMBED_MODEL_PATH
        self.bm25_path = BM25_MODEL_PATH
        self.index_path = INDEX_MODEL_PATH
        self.data_path = ABSTRACT_PATH
        self.model = None
        self.index = None
        self.bm25 = None
        self.encode = None
        self.encoded_query = None

    def init_model(self):
        self.model = SentenceTransformer(self.model_path)
        self.index = faiss.read_index(self.index_path)
        with open(self.bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)

    def init_data(self):
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.loc[self.df["Body"].notna()]
        self.training_data = self.df["Title"].values + self.df["Body"].values

    def encode_query(self, query):
        self.query = query
        self.encoded_query = self.model.encode([query])

    def search_query(self):
        if self.encoded_query is None:
            print("run model.encode_query(query) first")
            return None
        query_tokens = self.query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        ranked_indices = np.argsort(bm25_scores)[::-1]
        bm25_results = [(i, self.df.iloc[i], self.training_data[i]) for i in ranked_indices if bm25_scores[i] > 0]

        distances, indices = self.index.search(np.array(self.encoded_query, dtype=np.float32), k=2)
        faiss_results = [(i, self.df.iloc[i], self.training_data[i]) for i in indices[0]]
        ranked_results = reciprocal_rank_fusion(bm25_results, faiss_results)
        return [(df.iloc[i], abstract_bodies[i]) for i in ranked_results]
        return []


def test_model():
    test = TestModel()
    test.init_model()
    test.init_data()
    test.encode_query("test")
    test.search_query()



class AbstractSearcher:
    def __init__(self, *args, **kwargs):
        # DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
        # os.makedirs(DATA_DIR, exist_ok=True)
        self.ABSTRACT_PATH = ABSTRACT_PATH
        self.INDEX_MODEL_PATH = INDEX_MODEL_PATH
        self.BM25_MODEL_PATH = BM25_MODEL_PATH
        self.EMBED_MODEL_PATH = EMBED_MODEL_PATH
        self._trained = False
        self._df = None
        self._training_data = None
        self._bm25_model = None
        self._faiss_model = None
        self._index = None
        self._models_loaded = False
        self._query = None
        self._encoded_query = None
        self.load_data()

    def load_data(self):
        self._df = pd.read_csv(self.ABSTRACT_PATH)
        self._df = self._df.loc[self._df["Body"].notna()]
        self._training_data = self._df["Title"].values + self._df["Body"].values

    def train(self):
        print("Training model")
        training(self._df)
        self._trained = True
        self.load_models()

    def load_models(self, device="cpu"):
        with open(self.BM25_MODEL_PATH, "rb") as f:
            self._bm25_model = pickle.load(f)
        self._faiss_model = SentenceTransformer(self.EMBED_MODEL_PATH)
        self._index = faiss.read_index(self.INDEX_MODEL_PATH)
        self._models_loaded = True

    def query_data(self, query):
        self._query = query
        self.encode_query()
        if self._models_loaded:
            print("Searching abstracts")
            print("Query: ", query)
            print("Query type: ", type(query))
            query_tokens = query.lower().split()
            bm25_scores = self._bm25_model.get_scores(query_tokens)
            ranked_indices = np.argsort(bm25_scores)[::-1]
            bm25_results = [(i, self._df.iloc[i], self._training_data[i]) for i in ranked_indices if bm25_scores[i] > 0]
            print("Obtaining results by FAISS")
            distances, indices = self._index.search(np.array(self._encoded_query, dtype=np.float32), k=len(bm25_results))
            faiss_results = [(i, self._df.iloc[i], self._training_data[i]) for i in indices[0]]
            print("Results obtained by FAISS")
            ranked_results = reciprocal_rank_fusion(bm25_results, faiss_results)
            print("Results combined and ranked")
            return [(self._df.iloc[i], self._training_data[i]) for i in ranked_results]
        else:
            print("Models haven't been loaded, run load_models()")
            return None

    def encode_query(self):
        try:
            with Pool(processes=4) as pool:
                self._encoded_query = self._faiss_model.encode([self._query], pool=pool)
        except Exception as e:
            print(f"Error: {e}")
            self._encoded_query = []

def model_test():
    searcher = AbstractSearcher()
    searcher.load_models()
    return searcher.query_data("B7H3")

