from prompt import get_prompt
from utils.text_utils import init_bm25,init_embedding_corpus,load_metadata,retrieve,smooth_contexts
from constant import data_path, embedding_path, corpus_embedding
from utils.embed_corpus import load_embed_model


def load_context(data_path, corpus_embedding):
    meta_corpus = load_metadata(data_path)
    bm25 = init_bm25(meta_corpus)
    embedding_corpus = init_embedding_corpus(corpus_embedding)
    return bm25, embedding_corpus,meta_corpus

def gen_question(question, embed_model=None, corpus=None, bm25=None, embedding_corpus=None, topk=50):
    top_passages = retrieve(question, topk=50, bm25=bm25, embed_model=embed_model, corpus=corpus, corpus_embs=embedding_corpus,threshold=0.55)
    
    smoothed_contexts = ""
    
    if top_passages:
        smoothed_contexts = smooth_contexts(top_passages, corpus, word_window=60, n_sent=3)
    
    prompt = get_prompt(question, smoothed_contexts)
    #print(prompt)
    return prompt


if __name__ == "__main__":
    
    embedding_model = load_embed_model(embedding_path)
    
    bm25, embedding_corpus,meta_corpus = load_context(data_path,corpus_embedding)
    
    prompt = gen_question("Tôi muốn đi du lịch hà nội cuối tuần này ?", embed_model=embedding_model,bm25=bm25, corpus=meta_corpus, embedding_corpus=embedding_corpus,topk=10)
    print(prompt)
