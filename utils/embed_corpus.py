from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from tqdm import tqdm 
import numpy as np


def load_embed_model(model_path):
    model = SentenceTransformer(model_path).cuda()
    print("Done load embedding model")
    return model

def embed_corpus(embed_model, meta_corpus):
    segmented_corpus = [tokenize(example["passage"]) for example in tqdm(meta_corpus)]
    embeddings = embed_model.encode(segmented_corpus)
    embeddings /= np.linalg.norm(embeddings, axis=1)[:, np.newaxis]