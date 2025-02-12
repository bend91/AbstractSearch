# from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from rank_bm25 import BM25Okapi
import os
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
        self._encoded_query = self._faiss_model.encode([self._query])

def model_test():
    searcher = AbstractSearcher()
    searcher.load_models()
    return searcher.query_data("B7H3")



# Constants
# DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
# NLTK_DIR = os.path.join(os.path.dirname(__file__), "nltk_data")
# BM25_MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "bm25_model.pkl")
# INDEX_MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "abstracts.index")
# EMBED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "embed_model.pkl")
# ABSTRACT_PATH = os.path.join(os.path.dirname(__file__), "data", "Car_abstracts.csv")
# SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# os.makedirs(DATA_DIR, exist_ok=True)
# os.makedirs(NLTK_DIR, exist_ok=True)
# nltk.data.path = [NLTK_DIR]



# try:
#     nltk.download("punkt_tab", download_dir=NLTK_DIR, quiet=True)
#     nltk.download("stopwords", download_dir=NLTK_DIR, quiet=True)
# except LookupError:
#     raise LookupError("NLTK data download failed, Please check your internet connection")

# STOPWORDS = set(stopwords.words("english"))



# def search_abstracts(query, abstract_fp=ABSTRACT_PATH, top_k=2, use_embeddings=True):
#     print("Searching abstracts")
#     print("Query: ", query)
#     print("Query type: ", type(query))
#     query_tokens = query.lower().split()
#     df = pd.read_csv(abstract_fp)
#     df = df.loc[df["Body"].notna()]
#     abstract_bodies = df["Body"].values

#     try:
#         with open(BM25_MODEL_PATH, "rb") as f:
#             bm25 = pickle.load(f)
#     except FileNotFoundError:
#         raise FileNotFoundError("BM25 model has not been saved, re-train or change filename to bm25_model.pkl")
#     bm25_scores = bm25.get_scores(query_tokens)
#     ranked_indices = np.argsort(bm25_scores)[::-1]
#     bm25_results = [(i, df.iloc[i], abstract_bodies[i]) for i in ranked_indices if bm25_scores[i] > 0]
#     print("Results obtained by BM25")
#         # return {"method": "Keyword match (BM25)", "results": keyword_results}
#     try:
#         with open(EMBED_MODEL_PATH, "rb") as f:
#             embed_model = pickle.load(f)
#             # embed_model = SentenceTransformer(embed_model.config._name_or_path, device="cpu")
#     except FileNotFoundError:
#         raise FileNotFoundError("Embed model has not been saved, re-train or change filename to embed_model.pkl")
#     try:
#         index = faiss.read_index(INDEX_MODEL_PATH)
#     except FileNotFoundError:
#         raise FileNotFoundError("abstracts.index not found, if training has been completed check location or name of index file")
#     print("Embedding query")
#     query_embedding = embed_model.encode([query])
#     print("Obtaining results by FAISS")
#     distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k=len(bm25_results))
#     faiss_results = [(i, df.iloc[i], abstract_bodies[i]) for i in indices[0]]
#     print("Results obtained by FAISS")
#     ranked_results = reciprocal_rank_fusion(bm25_results, faiss_results)
#     print("Results combined and ranked")
#     return [(df.iloc[i], abstract_bodies[i]) for i in ranked_results]
        # return {"method": "Semantic Search (FAISS)", "results": faiss_results}
    # return {"method": "No results", "results": []}


# def bm25_training(df):
#     abstract_bodies = df["Body"].values

#     abstract_bodies = [preprocess_text(abstract, stopwords) for abstract in abstract_bodies]
#     print("training BM25Okapi")
#     bm25 = BM25Okapi(abstract_bodies)
#     with open(bm25_model_path, "wb") as f:
#         pickle.dump(bm25, f)


# def faiss_training(df):
#     abstract_bodies = df["Title"].values + df["Body"].values
#     print("encoding abstracts")
#     embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#     with open(EMBED_MODEL_PATH, "wb") as f:
#         pickle.dump(embed_model, f)
#     embeddings = embed_model.encode(abstract_bodies)
#     d = embeddings.shape[1]
#     print("indexing embeddings")
#     index = faiss.IndexFlatL2(d)
#     index.add(np.array(embeddings, dtype=np.float32))
#     print("writing indexes")
#     faiss.write_index(index, INDEX_MODEL_PATH)


# def training(df):
#     faiss_training(df)
#     bm25_training(df)


# def preprocess_text(text, stopwords=STOPWORDS):
#     return [word for word in word_tokenize(text.lower()) if word not in stopwords]





# def main(trained, abstract_fp):
#     df = pd.read_csv(abstract_fp)
#     df = df.loc[df["Body"].notna()]
#     if not trained:
#         training(df)


if __name__ == "__main__":
    print("main")
    # trained = True
    # print(sys.argv)
    # if len(sys.argv) > 1:
    #     if sys.argv[1] == "train":
    #         trained = False
    #         main(trained, ABSTRACT_PATH)
    #     else:
    #         query = sys.argv[1]
    #         results = search_abstracts(query)
    #         print("method", results["method"])
    #         for i, result in enumerate(results["results"], 1):
    #             print(i, result["Body"], '\n')
    # else:
    #     main(trained, ABSTRACT_PATH)
