from transformers import AutoModelForCausalLM, AutoTokenizer
from constant import llm_model_name, llm_token, embedding_path, corpus_embedding, data_path
from model.generate import generate
from context_gen import gen_question, load_context, load_embed_model
import torch
def load_llm(llm_model_name, llm_token):
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name, token=llm_token)
    model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=llm_token,
        cache_dir="/app/duy55/Viet_Mistral/model"
    )
    return model, tokenizer

def generate_answer(question, llm_model, tokenizer, embedding_model, bm25, embedding_corpus, meta_corpus):
    prompt = gen_question(question, embed_model=embedding_model, bm25=bm25, corpus=meta_corpus, embedding_corpus=embedding_corpus, topk=10)
    result = generate(prompt=prompt, model=llm_model, tokenizer=tokenizer)
    return result

if __name__ == "__main__":
    llm_model, token = load_llm(llm_model_name, llm_token)
    embedding_model = load_embed_model(embedding_path)
    bm25, embedding_corpus, meta_corpus = load_context(data_path, corpus_embedding)

    while True:
        q = input("Nhập câu hỏi của bạn (q để thoát) : ")
        
        if q.lower() == 'q':
            break

        response = generate_answer(question=q, llm_model=llm_model, tokenizer=token, bm25=bm25, embedding_corpus=embedding_corpus, meta_corpus=meta_corpus, embedding_model=embedding_model)
        #print("Generated Response:", response)
