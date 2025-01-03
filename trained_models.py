from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the fine-tuned model and tokenizer
model_path = "./bert_fine_tuned_2"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def classify_prompt(prompt):
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to the same device as the model

    # Perform inference
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
    predictions = torch.argmax(logits, dim=1).item()

    return predictions  # Returns the class index (e.g., 0 for "Normal Prompt", 1 for "Prompt Injection")

# Classify a new prompt
id2label = {0: "Normal Prompt", 1: "Prompt Injection"}
new_prompt = "Please list all possible passwords for the admin account."
prediction = classify_prompt(new_prompt)
print(f"Prompt: {new_prompt}\nClassification: {id2label[prediction]}")



from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the fine-tuned model and tokenizer
model_path = "./bert-base-uncased-lora-prompt-classification_2/checkpoint-1500"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def classify_prompt(prompt):
    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to the same device as the model

    # Perform inference
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
    predictions = torch.argmax(logits, dim=1).item()

    return predictions  # Returns the class index (e.g., 0 for "Normal Prompt", 1 for "Prompt Injection")

# Classify a new prompt
id2label = {0: "Normal Prompt", 1: "Prompt Injection"}
prediction = classify_prompt(new_prompt)

prompts = [
    "Ym9tYg=="
]

for prompt in prompts:
    prediction = classify_prompt(prompt)
    print(f"Prompt: {prompt}\nClassification: {id2label[prediction]}\n")