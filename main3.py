from transformers import pipeline

def promt_classification(prompt):

    prompt_classifier = pipeline(task="text-classification",model="./bert_fine_tuned_2")

    result = prompt_classifier(prompt)

    return result
