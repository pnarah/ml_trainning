import json

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import DataLoader, Dataset

# Initialize tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")


with open('augmented_training_data_30var.json', 'r') as f:
    training_data = json.load(f)

# Prepare training data
input_texts = [data["input_text"] for data in training_data]

target_texts = []
for data in training_data:
    api_request = data.get("api_request", {})  # Get api_request or empty dict if not present
    params = api_request.get("params", {}) # Get params or empty dict if not present
    headers = api_request.get("headers", {})
    data = api_request.get("data", {})
    # If params is a dictionary, format it as a string. Otherwise, use the existing value.
    params_str = str(params) if isinstance(params, dict) else params
    headers_str = str(headers) if isinstance(headers, dict) else headers
    data_str = str(data) if isinstance(data, dict) else data

    target_texts.append(
        f'endpoint: {api_request.get("endpoint", "")} method: {api_request.get("method", "")} params: {params_str} headers: {headers_str} data: {data_str}'
    )

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
# # Training loop
# model.train()
# for epoch in range(3):  # Train for 3 epochs
#     for batch in loader:
#         optimizer.zero_grad()
#         outputs = model(input_ids=batch["input_ids"],
#                         attention_mask=batch["attention_mask"],
#                         labels=batch["labels"])
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#     print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


# Initialize lists to store training history
loss_history = []
accuracy_history = []

# Training loop
model.train()
for epoch in range(3):  # Train for 3 epochs
    epoch_loss = 0
    correct_predictions = 0
    total_predictions = 0
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        # Accumulate loss for the epoch
        epoch_loss += loss.item()
        # Calculate accuracy
        # Assuming outputs.logits provides the raw predictions
        predictions = torch.argmax(outputs.logits, dim=-1)
        correct_predictions += (predictions == batch["labels"]).sum().item()
        total_predictions += batch["labels"].numel()
        print(f"epoch_loss {epoch_loss} correct_predictions {correct_predictions} total_predictions {total_predictions}")
    # Calculate average loss and accuracy for the epoch
    avg_epoch_loss = epoch_loss / len(loader)
    epoch_accuracy = correct_predictions / total_predictions
    loss_history.append(avg_epoch_loss)
    accuracy_history.append(epoch_accuracy)
    print(f"Epoch {epoch + 1}, Average Loss: {avg_epoch_loss}, Accuracy: {epoch_accuracy}")

# At the end of training, loss_history and accuracy_history contain the average loss and accuracy for each epoch


# Save the pyTorch model
torch.save(model.state_dict(), "apipathmodel1.pth")

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
import re

# Function to parse each generated text
def parse_generated_text(text):
    # Regular expressions to extract relevant parts
    endpoint_pattern = r"endpoint:\s*(\S+)"
    method_pattern = r"method:\s*(\w+)"
    headers_pattern = r"headers:\s*(.*?)(?:data:|$)"
    email_pattern = r"'email':\s*'([^']*)'"
    team_id_pattern = r"'teamId':\s*(\d+)"
    param_pattern = r"params:\s*(.*?)(?:headers|data|$)"

    # Extract values using regular expressions, with fallback defaults
    endpoint = re.search(endpoint_pattern, text).group(1)
    method = re.search(method_pattern, text).group(1)

    headers_match = re.search(headers_pattern, text)
    headers_text = headers_match.group(1) if headers_match else ""
    params_match = re.search(param_pattern, text)
    params_text = params_match.group(1) if params_match else ""

    # Parse headers if present
    headers = {}
    if "'accept':" in headers_text and "'Content-Type':" in headers_text:
        headers = {
            "accept": re.search(r"'accept':\s*'([^']+)'", headers_text).group(1),
            "Content-Type": re.search(r"'Content-Type':\s*'([^']+)'", headers_text).group(1)
        }

    # Extract specific data fields if present
    email_match = re.search(email_pattern, text)
    email = email_match.group(1) if email_match else None
    team_id_match = re.search(team_id_pattern, text)
    team_id = int(team_id_match.group(1)) if team_id_match else None

    # Parse params if present
    params = {}
    if "'dc_id':" in params_text:
        params["dc_id"] = re.search(r"'dc_id':\s*'(\d+)'", params_text).group(1)

    # Construct the JSON structure based on extracted values
    formatted_output = {
        "endpoint": endpoint,
        "method": method,
        # "headers": headers,
        # "data": {}
    }

    # Add data or params based on extracted values
    if email and team_id is not None:
        formatted_output["data"] = {
            "email": email,
            "teamId": team_id
        }
    if "dc_id" in params:
        formatted_output["params"] = params  # For GET, we can treat it as params
    if headers:
        formatted_output["headers"] = headers

    return formatted_output

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
# test_input = "arrive detail of team thirty from SPOG"
# test_input = "draw datum center from SPOG with datum center id six"
test_input="add user with e-mail id pnarah3@maildummy.com and team id 9"
generated_text = generate_api_request(test_input)
print("Generated Text:", generated_text)
api_request = parse_generated_text(generated_text)
print("Parsed API Request:", json.dumps(api_request, indent=2))
# response = execute_api_request(api_request)
# print("API Response:", response)


# Clear cache
# rm -rf ~/.cache/huggingface/hub