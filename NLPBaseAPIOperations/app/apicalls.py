from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Initialize FastAPI app
app = FastAPI(title="apicalls")
import os
print("Current working directory:", os.getcwd())

# Define model architecture (using T5 as an example based on the training script)
class MyModel:
    def __init__(self, model_path="/Users/pnarah/git-pn/ml_trainning/NLPBaseAPIOperations/app/apipathmodel1.pth"):
        # Load the pre-trained model architecture
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set to evaluation mode
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")

    def predict(self, input_text):
        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            # Generate output from model
            outputs = self.model.generate(inputs.input_ids, attention_mask=inputs.attention_mask)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Instantiate the model
model_instance = MyModel("/NLPBaseAPIOperations/app/apipathmodel1.pth")

# Define input data model for FastAPI
class InputData(BaseModel):
    input_text: str


@app.post("/predict")
async def predict(data: InputData):
    try:
        # Perform inference
        prediction = model_instance.predict(data.input_text)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
