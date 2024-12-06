import os
import torch
# from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Define model loading function
def model_fn(model_dir):
    """Load the PyTorch model from the model directory."""
    model_path = os.path.join(model_dir, 'spogmodel1.pth')
    model_name = "google/long-t5-tglobal-base"
    # tokenizer = T5Tokenizer.from_pretrained("t5-base")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.model_max_length = 1024
    # model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # model.config.max_length = 1024
    model.load_state_dict(torch.load(model_path))
    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return {"model": model, "tokenizer": tokenizer}


# Define input data processing function
def input_fn(input_data, content_type):
    """Deserialize JSON input into a format usable by the model."""
    if content_type == "application/json":
        import json
        data = json.loads(input_data)
        if "inputs" not in data:
            raise ValueError("Expected 'inputs' field in input JSON")
        return data["inputs"]
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


# Define inference logic
def predict_fn(input_data, model_and_tokenizer):
    """Perform inference on the input data using the loaded model."""
    # model = model_data["model"]
    # tokenizer = model_data["tokenizer"]
    model = model_and_tokenizer["model"]
    tokenizer = model_and_tokenizer["tokenizer"]

    # Tokenize input text
    inputs = tokenizer(input_data, return_tensors="pt", truncation=True, padding=True)

    # Generate predictions
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=600,  # Adjust max_length based on your needs
        num_beams=4,  # Number of beams for beam search
        early_stopping=True,
        output_scores=True,
        return_dict_in_generate=True
    )
    # Get confidence score
    scores = outputs.sequences_scores[0].item()
    print(f"outputs sequences_scores : {scores}")

    # Set a confidence threshold
    threshold = -0.03
    if scores < threshold:
        return "Request couldnot be translated into action. Please try again."

    decoded_output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    return decoded_output


# Define output data processing function
def output_fn(prediction, accept):
    """Serialize the predictions to JSON."""
    if accept == "application/json":
        import json
        print(f"output_fn prediction : {prediction}")
        return json.dumps({"result": prediction}), "application/json"
    else:
        raise ValueError(f"Unsupported accept type: {accept}")