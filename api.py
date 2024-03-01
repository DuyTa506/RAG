from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from context_gen import load_context, load_embed_model
from constant import llm_model_name, llm_token, embedding_path, corpus_embedding, data_path
import torch
from main import generate_answer, generate_answer_with_api
app = FastAPI()

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



llm_model, token = load_llm(llm_model_name, llm_token)
embedding_model = load_embed_model(embedding_path)
bm25, embedding_corpus, meta_corpus = load_context(data_path, corpus_embedding)

class Question(BaseModel):
    question: str

@app.post("/generate_answer")
def generate_answer_endpoint(question_data: Question):
    try:

        question = question_data.question

        response = generate_answer_with_api(question=question, llm_model=llm_model, tokenizer=token,
                                   bm25=bm25, embedding_corpus=embedding_corpus, meta_corpus=meta_corpus,
                                   embedding_model=embedding_model)

        return JSONResponse(content={"response": response}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



