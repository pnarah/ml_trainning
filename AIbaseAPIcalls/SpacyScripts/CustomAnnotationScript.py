# example Use libraries like spaCy’s annotation tool to label text data.
#
# pip install spacy
# python -m spacy download en_core_web_sm


import spacy
from spacy.tokens import DocBin
import json

# Load SpaCy model
nlp = spacy.blank("en")  # Start with a blank model for custom annotations

# Sample sentences for annotation
texts = [
    "Add team to SPOG with team name Apple with password Hellowme",
    "Create a new team in SPOG with team name Banana and password Secure123",
]

# Define the labels you want to annotate
LABELS = ["TEAM_NAME", "PASSWORD"]

# Function for annotating data
def annotate_texts(texts):
    annotations = []
    for text in texts:
        print(f"\nText: {text}")
        doc = nlp.make_doc(text)

        entities = []
        print("Please enter the start and end positions for each entity.")
        print(f"Possible labels: {LABELS}")

        while True:
            try:
                # User inputs the start and end position of the entity
                start = int(input("Start position (or -1 to finish): "))
                if start == -1:
                    break
                end = int(input("End position: "))
                label = input(f"Label (choose from {LABELS}): ")

                # Validate the label
                if label not in LABELS:
                    print("Invalid label! Try again.")
                    continue

                # Add entity data
                entities.append((start, end, label))
                print(f"Annotated entity '{text[start:end]}' as {label}")
            except ValueError:
                print("Invalid input. Please enter numbers for start/end positions.")

        annotations.append((text, {"entities": entities}))

    return annotations

# Annotate texts
annotated_data = annotate_texts(texts)

# Save annotated data to a JSON file for future use
with open("annotated_data.json", "w") as f:
    json.dump(annotated_data, f)

print("\nAnnotations saved to annotated_data.json")
