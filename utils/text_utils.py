import copy
import pickle
import unicodedata as ud
import re
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from tqdm.notebook import tqdm
import string
import numpy as np

from pyvi.ViTokenizer import tokenize
def split_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.lower().split()
    words = [word for word in words if len(word.strip()) > 0]
    return words

## initiate BM25 retriever
def init_bm25(meta_corpus):
    tokenized_corpus = [split_text(doc["passage"]) for doc in tqdm(meta_corpus)]
    bm25 = BM25Okapi(tokenized_corpus)
    print("Done init bm25!!")
    return bm25
def init_embedding_corpus(path):
    with open(path, 'rb') as f:
        corpus_embs = pickle.load(f)
    print("Done embedded corpus! ")
    return corpus_embs

from datasets import load_dataset
def load_metadata(path):

    meta_corpus = load_dataset(
        "json",
        data_files=path,
        split="train"
    ).to_list()
    print("Done load meta_corpus !")
    return meta_corpus


from copy import deepcopy
def retrieve(question,  embed_model, bm25,corpus,corpus_embs,topk=70,threshold = 0.6):
    """
    Get most relevant chunks to the question using combination of BM25 and semantic scores.
    """
    ## initialize query for each retriever (BM25 and semantic)
    tokenized_query = split_text(question)
    segmented_question = tokenize(question)
    question_emb = embed_model.encode([segmented_question])
    question_emb /= np.linalg.norm(question_emb, axis=1)[:, np.newaxis]

    ## get BM25 and semantic scores
    bm25_scores = bm25.get_scores(tokenized_query)
    semantic_scores = question_emb @ corpus_embs.T
    semantic_scores = semantic_scores[0]

    ## update chunks' scores. 
    max_bm25_score = max(bm25_scores)
    min_bm25_score = min(bm25_scores)
    def normalize(x):
        return (x - min_bm25_score + 0.1) / \
        (max_bm25_score - min_bm25_score + 0.1)
        
    corpus_size = len(corpus)
    for i in range(corpus_size):
        corpus[i]["bm25_score"] = bm25_scores[i]
        corpus[i]["bm25_normed_score"] = normalize(bm25_scores[i])
        corpus[i]["semantic_score"] = semantic_scores[i]

    ## compute combined score (BM25 + semantic)
    for passage in corpus:
        passage["combined_score"] = passage["bm25_normed_score"] * 0.4 + \
                                    passage["semantic_score"] * 0.6

    ## sort passages by the combined score
    sorted_passages = sorted(corpus, key=lambda x: x["combined_score"], reverse=True)
    filtered_passages = [passage for passage in sorted_passages if passage["combined_score"] > threshold]
    return filtered_passages[:topk]


from copy import deepcopy
from underthesea import sent_tokenize
def extract_consecutive_subarray(numbers):
    subarrays = []
    current_subarray = []
    for num in numbers:
        if not current_subarray or num == current_subarray[-1] + 1:
            current_subarray.append(num)
        else:
            subarrays.append(current_subarray)
            current_subarray = [num]

    subarrays.append(current_subarray)  # Append the last subarray
    return subarrays
    
def merge_contexts(passages):
    passages_sorted_by_id = sorted(passages, key=lambda x: x["id"], reverse=False)
    # psg_texts = [x["passage"].strip("Title: ").strip(x["title"]).strip() 
    #              for x in passages_sorted_by_id]
    
    psg_ids = [x["id"] for x in passages_sorted_by_id]
    consecutive_ids = extract_consecutive_subarray(psg_ids)

    merged_contexts = []
    consecutive_psgs = []
    b = 0
    for ids in consecutive_ids:
        psgs = passages_sorted_by_id[b:b+len(ids)]
        psg_texts = [x["passage"].strip("Title: ").strip(x["title"]).strip() 
                     for x in psgs]
        merged = f"Title: {psgs[0]['title']}\n\n" + " ".join(psg_texts)
        b = b+len(ids)
        merged_contexts.append(dict(
            title=psgs[0]['title'], 
            passage=merged,
            score=max([x["combined_score"] for x in psgs]),
            merged_from_ids=ids
        ))
    return merged_contexts

def discard_contexts(passages):
    sorted_passages = sorted(passages, key=lambda x: x["score"], reverse=False)
    if len(sorted_passages) == 1:
        return sorted_passages
    else:
        shortened = deepcopy(sorted_passages)
        for i in range(len(sorted_passages) - 1):
            current, next = sorted_passages[i], sorted_passages[i+1]
            if next["score"] - current["score"] >= 0.15:
                shortened = sorted_passages[i+1:]
        return shortened

def expand_context(passage,meta_corpus, word_window=60, n_sent=3):
    # psg_id = passage["id"]
    merged_from_ids = passage["merged_from_ids"]
    title = passage["title"]
    prev_id = merged_from_ids[0] - 1
    next_id = merged_from_ids[-1] + 1
    strip_title = lambda x: x["passage"].strip(f"Title: {x['title']}\n\n")
    
    texts = []
    if prev_id in range(0, len(meta_corpus)):
        prev_psg = meta_corpus[prev_id]
        if prev_psg["title"] == title: 
            prev_text = strip_title(prev_psg)
            # prev_text = " ".join(prev_text.split()[-word_window:])
            prev_text = " ".join(sent_tokenize(prev_text)[-n_sent:])
            texts.append(prev_text)
            
    texts.append(strip_title(passage))
    
    if next_id in range(0, len(meta_corpus)):
        next_psg = meta_corpus[next_id]
        if next_psg["title"] == title: 
            next_text = strip_title(next_psg)
            # next_text = " ".join(next_text.split()[:word_window])
            next_text = " ".join(sent_tokenize(next_text)[:n_sent])
            texts.append(next_text)

    expanded_text = " ".join(texts)
    expanded_text = f"Title: {title}\n{expanded_text}"
    new_passage = deepcopy(passage)
    new_passage["passage"] = expanded_text
    return new_passage

def expand_contexts(passages,meta_corpus, word_window=60, n_sent=3):
    new_passages = [expand_context(passage,meta_corpus,word_window=word_window,n_sent=n_sent) for passage in passages]
    return new_passages
    
def collapse(passages):
    new_passages = deepcopy(passages)
    titles = {}
    for passage in new_passages:
        title = passage["title"]
        if not titles.get(title):
            titles[title] = [passage]
        else:
            titles[title].append(passage)
    best_passages = []
    for k, v in titles.items():
        best_passage = max(v, key= lambda x: x["score"])
        best_passages.append(best_passage)
    return best_passages


def smooth_contexts(passages,meta_corpus,word_window=60,n_sent=3):
    """Make the context fed to the LLM better.
    Args:
        passages (list): Chunks retrieved from BM25 + semantic retrieval. 
        
    Returns:
        list: List of whole paragraphs, usually will be more relevant to the initital question.
    """
    # 1. If consecutive chunks are rertieved, merge them into one big chunk to ensure the continuity.
    merged_contexts = merge_contexts(passages)
    # 2. A heuristic to discard irrelevevant contexts. 
    # It seems to be better to only keep what are elevant so that the model can focus.
    # Also this reduce #tokens LLM has to read.
    shortlisted_contexts = discard_contexts(merged_contexts)
    # 3. Another heuristic. this step is to take advantage of long context understanding of the LLM.
    # In many cases, the retrieved passages are just consecutive words, not a comprehensive paragraph.
    # This is to expand the passage to the whole paragraph that surrounds it. 
    # My intuition about this is that whole paragraph will add necessary and relevant information.
    expanded_contexts = expand_contexts(shortlisted_contexts,meta_corpus=meta_corpus,word_window=word_window,n_sent=n_sent)
    # 4. Now after all the merging and expanding, if what are left for us is more than one paragraphs
    # from the same wiki page, then we will only take paragraph with highest retrieval score.
    collapsed_contexts = collapse(expanded_contexts)
    return collapsed_contexts