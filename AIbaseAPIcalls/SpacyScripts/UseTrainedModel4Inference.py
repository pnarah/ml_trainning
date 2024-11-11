# Step 4: Train the SpaCy NER Model with the Annotated Data
# Use SpaCy’s CLI to train a model on the annotated data. First, create a config.cfg file with SpaCy’s quickstart CLI:
#
#     python - m spacy init config config.cfg - -lang en - -pipeline ner
#
# Then, update the config.cfg to point to your annotated data, and run the training:
#
#     python -m spacy train config.cfg --output ./output --paths.train ./training_data.spacy --paths.dev ./training_data.spacy


import spacy

# Load the trained model
nlp_ner = spacy.load("./output/model-best")

# Test the model
test_text = "Add team to SPOG with team name Apple with password Hellowme"
doc = nlp_ner(test_text)

# Print extracted entities
for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")
