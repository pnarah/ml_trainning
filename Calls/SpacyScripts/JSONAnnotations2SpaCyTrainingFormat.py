# Once you have annotated_data.json, convert this into SpaCy’s binary format (.spacy) for model training.
import spacy
from spacy.tokens import DocBin

# Load blank SpaCy model
nlp = spacy.blank("en")

# Load annotated data
with open("annotated_data.json", "r") as f:
    training_data = json.load(f)

# Convert training data to DocBin format
doc_bin = DocBin()
for text, annotations in training_data:
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in annotations["entities"]:
        span = doc.char_span(start, end, label=label)
        if span is not None:
            ents.append(span)
    doc.ents = ents  # Set entities for the document
    doc_bin.add(doc)

# Save the data in SpaCy binary format
doc_bin.to_disk("training_data.spacy")
print("Training data saved to training_data.spacy")
