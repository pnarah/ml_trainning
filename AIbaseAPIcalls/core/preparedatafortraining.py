import json

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import DataLoader, Dataset

# Initialize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")


with open('augmented_training_data.json', 'r') as f:
    training_data = json.load(f)

# Prepare training data
input_texts = [data["input_text"] for data in training_data]

target_texts = []
for data in training_data:
    api_request = data.get("api_request", {})  # Get api_request or empty dict if not present
    params = api_request.get("params", {}) # Get params or empty dict if not present
    # If params is a dictionary, format it as a string. Otherwise, use the existing value.
    params_str = str(params) if isinstance(params, dict) else params
    target_texts.append(
        f'endpoint: {api_request.get("endpoint", "")} method: {api_request.get("method", "")} params: {params_str}'
    )

# target_texts = [
#     f'endpoint: {data["api_request"]["endpoint"]} method: {data["api_request"]["method"]} params: {data["api_request"]["params"]}'
#     for data in training_data
# ]

# Tokenize the input and target text
input_encodings = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
target_encodings = tokenizer(target_texts, padding=True, truncation=True, return_tensors="pt")


# Create a custom dataset
class APIDataset(Dataset):
    def __init__(self, input_encodings, target_encodings):
        self.input_encodings = input_encodings
        self.target_encodings = target_encodings

    def __len__(self):
        return len(self.input_encodings.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_encodings.input_ids[idx],
            "attention_mask": self.input_encodings.attention_mask[idx],
            "labels": self.target_encodings.input_ids[idx]
        }

# Instantiate dataset and dataloader
dataset = APIDataset(input_encodings, target_encodings)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
model.train()
for epoch in range(3):  # Train for 3 epochs
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")



# Sample inference function
def generate_api_request(input_text):
    model.eval()
    with torch.no_grad():
        # Tokenize input and generate output
        input_encoding = tokenizer(input_text, return_tensors="pt")
        generated_ids = model.generate(input_encoding.input_ids, max_length=50)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

# Test the model on new input
test_input = "retrieve user info for user_id 789"
print("Generated API Request:", generate_api_request(test_input))

# ===============
import json
import requests


# Simple parser function (depends on model output format)
def parse_generated_text(generated_text):
    # Split and parse based on known format
    parts = generated_text.split(" ")
    endpoint = parts[1]
    method = parts[3]
    params_str = " ".join(parts[5:])
    params = json.loads(params_str.replace("'", "\""))  # Convert single quotes to double quotes for JSON parsing
    return {
        "endpoint": endpoint,
        "method": method,
        "params": params
    }


# Function to execute the API call
def execute_api_request(api_request):
    base_url = "https://api.example.com"
    url = base_url + api_request["endpoint"]

    # Use appropriate method
    if api_request["method"] == "GET":
        response = requests.get(url, params=api_request["params"])
    elif api_request["method"] == "POST":
        response = requests.post(url, json=api_request["params"])
    else:
        raise ValueError("Unsupported HTTP method")

    # Handle response
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": response.status_code, "message": response.text}


# Execute the generated request
generated_text = generate_api_request(test_input)
api_request = parse_generated_text(generated_text)
print("Parsed API Request:", json.dumps(api_request, indent=2))
# response = execute_api_request(api_request)
# print("API Response:", response)
