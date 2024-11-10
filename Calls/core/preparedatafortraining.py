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
target_texts = [
    f'endpoint: {data["api_request"]["endpoint"]} method: {data["api_request"]["method"]} params: {data["api_request"]["params"]}'
    for data in training_data
]

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
