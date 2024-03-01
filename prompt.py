prompt_template = (
    "### System:\n"
    "{system_prompt}.\n\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

#def history_buffer(n = 1): 
    #from langchain.memory import ConversationBufferWindowMemory(k=n)
    #memory.save_context({"input": "not much you"}, {"output": "not much"})

def get_prompt(question, contexts):
    if not contexts:
        system_prompt = (
            "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.\n"
            "Câu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực.\n"
            "Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trả lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch.\n"
            #"Vui lòng ghi nhớ câu hỏi trước đó của người dùng để tạo thành chuỗi hội thoại nếu cần"
        )
        instruction = "Please answer the following question in a friendly and informative manner.\n"
        input_text = f"Câu hỏi: {question}\nHãy trả lời câu hỏi trên một cách chi tiết và đầy đủ.\n"
        prompt = prompt_template.format(
            system_prompt=system_prompt,
            instruction=instruction,
            input=input_text,
            output=''
        )
    else :
        context = "\n\n".join([f"Thông tin cung cấp [{i+1}]: {x['passage']}" for i, x in enumerate(contexts)])
        system_prompt = (
            "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.\n"
            "Câu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực.\n"
            "Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trả lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch.\n"
            #"Vui lòng ghi nhớ câu hỏi trước đó của người dùng để tạo thành chuỗi hội thoại nếu cần"
        )
        instruction = "Please answer the following travel-related question in a friendly and informative manner. Utilize both retrieved context and your knowledge to provide the most relevant information. If the question is not directly covered by the context, use your travel expertise to generate a helpful response.\n\n"
        input_text = f"Dựa vào một số ngữ cảnh được cho dưới đây, trả lời câu hỏi ở cuối.\n\{context}\n\ Câu hỏi: {question}\nHãy trả lời chi tiết và đầy đủ.\nNếu nhận thấy câu hỏi không có trong ngữ cảnh hoặc người dùng không cung cấp ngữ cảnh, hãy sử dụng tri thức vốn có của bạn và trả lời chi tiết\n "
        prompt = prompt_template.format(
            system_prompt=system_prompt,
            instruction=instruction,
            input=input_text,
            output=''
        )
    return prompt
