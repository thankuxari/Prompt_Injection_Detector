from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
import torch

def generate_pet_name(animal_type):
    # Load the model and tokenizer
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  
        device_map="auto"           
    )

    # Create a Hugging Face pipeline
    hf_pipeline = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

    # Wrap the pipeline in a LangChain-compatible LLM
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=["animal_type"],
        template="I have a {animal_type} and I want a cool name for it. Suggest me five different names."
    )

    # Create the LangChain LLMChain
    name_chain = LLMChain(llm=llm, prompt=prompt_template)

    # Run the chain
    response = name_chain.run({"animal_type": animal_type})
    
    return response


