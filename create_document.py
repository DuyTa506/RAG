from underthesea import sent_tokenize
import os
from tqdm import tqdm

def split_text_into_chunks(text, chunk_size=1000, window_size=50):
    """Split a long text into multiple chunks (passages) with managable sizes.
    
    Args:
        chunk_size (int): Maximum size of a chunk.
        window_size (int): Decide how many words are overlapped between two consecutive chunks. Basically #overlapped_words = chunk_size - window_size.
    Returns:
        str: Multiple chunks of text splitted from initial document text.
    """
    words = text.split()
    num_words = len(words)
    chunks = []
    start_idx = 0

    while True:
        end_idx = start_idx + chunk_size
        chunk = " ".join(words[start_idx:end_idx])
        chunks.append(chunk)
        if end_idx >= num_words:
            break
        start_idx += window_size

    return chunks

def get_corpus(data_dir="data/data_raw10k/"):
    """Transform a corpus of documents into a corpus of passages.
    
    Args:
        data_dir (str): directory that contains .txt files, each file contains text content of a wikipedia page.
    Returns:
        str: A corpus of chunks splitted from multiple initial documents. Each chunk will contain information about (id, title, passage)
    """
    corpus = []
    meta_corpus = []
    filenames = os.listdir(data_dir)
    filenames = sorted(filenames)
    
    _id = 0
    docs = {}
    for filename in tqdm(filenames):
        filepath = data_dir + filename
        title = filename.strip(".txt")
        with open(filepath, "r") as f:
            text = f.read()
            docs[title] = text
            text = text.lstrip(title).strip()

            # No overlap.
            chunks = split_text_into_chunks(text, chunk_size=150, window_size=150)
            chunks = [f"Title: {title}\n\n{chunk}" for chunk in chunks]
            meta_chunks = [{
                "title": title,
                "passage": chunks[i],
                "id": _id + i,
                "len": len(chunks[i].split())
            } for i in range(len(chunks))]
            _id += len(chunks)
            corpus.extend(chunks)
            meta_corpus.extend(meta_chunks)
    return meta_corpus


def create_corpus(data_dir): 
    return get_corpus(data_dir)


if __name__ == "__main__": 
    data_dir = "/app/duy55/Viet_Mistral/NCKH_2024_data/crawled/"
    meta_corpus = create_corpus(data_dir)
    print(meta_corpus[2000])
    