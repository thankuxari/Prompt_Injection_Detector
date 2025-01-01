from transformers import pipeline 

def answer() : 
    text_generator = pipeline(task="text-generation", model="HuggingFaceTB/SmolLM2-1.7B-Instruct")

    messages = [
        {"role": "user", "content": "What is the capital of Greece?"},
    ]

    result = text_generator(messages)
    
    return result

print(answer())