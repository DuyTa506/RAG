from transformers import GenerationConfig, TextStreamer
import torch


def remove_special_tokens(text, tokenizer):
    special_tokens = set(tokenizer.all_special_tokens)
    tokens = tokenizer.tokenize(text)
    filtered_tokens = [token for token in tokens if token not in special_tokens]
    filtered_text = tokenizer.convert_tokens_to_string(filtered_tokens)

    return filtered_text

def generate(prompt, tokenizer, model, max_new_tokens=1024):
    """Text completion with a given prompt. In other words, give an answer to your question.
    Args:
        prompt (str): Basically <instruction> + <question> + <retrieved_context>
        model (PreTrainedModel): The language model for generation.
        tokenizer (PreTrainedTokenizer): Tokenizer for processing input and output.
        max_new_tokens (int): Maximum number of tokens to generate.
    Returns:
        str: An answer to the question within the prompt.
    """
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    model.eval()
    with torch.no_grad():
        generation_config = GenerationConfig(
            repetition_penalty=1.13,
            max_new_tokens=max_new_tokens,
            temperature=1,
            top_p=0.3,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
        )
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        generated = model.generate(
            inputs=input_ids,
            generation_config=generation_config,
            streamer=streamer,
        )

    gen_tokens = generated["sequences"].cpu()[:, len(input_ids[0]):]
    output = tokenizer.batch_decode(gen_tokens, skip_special_tokens =True)[0]
    output = output.split(tokenizer.eos_token)[0]
    return output.strip()

def generate_with_api(prompt, tokenizer, model, max_new_tokens=512):
    """Text completion with a given prompt. In other words, give an answer to your question.
    Args:
        prompt (str): Basically <instruction> + <question> + <retrieved_context>
        model (PreTrainedModel): The language model for generation.
        tokenizer (PreTrainedTokenizer): Tokenizer for processing input and output.
        max_new_tokens (int): Maximum number of tokens to generate.
    Returns:
        str: An answer to the question within the prompt.
    """
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    model.eval()
    
    with torch.no_grad():
        generation_config = GenerationConfig(
            repetition_penalty=1.13,
            max_new_tokens=max_new_tokens,
            temperature=1,
            top_p=0.3,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
        )
        
        generated = model.generate(
            inputs=input_ids,
            generation_config=generation_config,
            max_length=input_ids.shape[1] + max_new_tokens,
            num_beams=5,  # You can adjust the number of beams for beam search
        )

    gen_tokens = generated[0].cpu()
    output = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]
    output = output.split(tokenizer.eos_token)[0]
    return output.strip()
