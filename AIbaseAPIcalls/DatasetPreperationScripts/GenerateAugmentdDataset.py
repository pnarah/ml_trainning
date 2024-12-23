import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')  # Additional language resources

from nltk.corpus import wordnet as wn
import random
import json


def generate_variations(input_text, num_variations=10):
    words = input_text.split()
    variations = set()  # Using a set to avoid duplicate sentences

    # For each word, find synonyms and create variations
    for _ in range(num_variations):
        new_sentence = []
        for word in words:
            # Get synonyms for words that are likely action verbs or nouns
            synonyms = wn.synsets(word, lang='eng')
            if synonyms:
                # Choose a random synonym's name, if available
                syn = random.choice(synonyms).lemma_names()[0]
                new_word = syn.replace('_', ' ') if syn != word else word
                new_sentence.append(new_word)
            else:
                new_sentence.append(word)
        # Join words to form a sentence and add to variations set
        variations.add(" ".join(new_sentence))

    # Ensure the original sentence is included
    variations.add(input_text)
    return list(variations)


def generate_type1_variations(template_data):
    generated_data = []
    # Loop over each template entry
    input_text_template = template_data["input_text_template"]
    endpoint_template = template_data["endpoint_template"]
    method = template_data["method"]
    variables = template_data["variables"]

    key, values = list(variables.items())[0]
    # Loop over all possible combinations of variables
    for value in values:
        # Substitute the single variable in the template
        input_text = input_text_template.format(**{key: value})
        endpoint = endpoint_template.format(**{key: value})

        form_data = {
            "input_text": input_text,
            "api_request": {
                "endpoint": endpoint,
                "method": method
            }
        }

        if 'headers' in template_data:
            form_data['api_request']['headers'] = template_data['headers']

        if 'params' in template_data:
            form_data['api_request']['params'] = template_data['params']
        # Add the populated entry
        generated_data.append(form_data)

    return generated_data

def generate_type2_parameter_variations(template_data):
    # print(template_data["input_text_template"])
    generated_data = []
    keys = template_data["variables"].keys()
    if len(keys) == 1 and "team_name" in keys:
        # print(template_data["variables"]["team_name"])
        for team_name in template_data["variables"]["team_name"]:
            # Replace placeholders with the actual values
            populated_request = {
                "input_text": template_data["input_text_template"].format(team_name=team_name),
                "api_request": {
                    "endpoint": template_data["api_request"]["endpoint"],
                    "method": template_data["api_request"]["method"],
                    "headers": template_data["api_request"]["headers"],
                    "data": {
                        "name": team_name,
                        "is_customer": template_data["api_request"]["data"]["is_customer"],
                    }
                }
            }
            # print(populated_request)
    else:
        for email, team_id in zip(template_data["variables"]["email"], template_data["variables"]["teamId"]):
            # Replace placeholders with the actual values
            populated_request = {
                "input_text": template_data["input_text_template"].format(email=email, teamId=team_id),
                "api_request": {
                    "endpoint": template_data["api_request"]["endpoint"],
                    "method": template_data["api_request"]["method"],
                    "headers": template_data["api_request"]["headers"],
                    "data": {
                        "email": email,
                        "teamId": team_id
                    }
                }
            }
            # print(populated_request)
    generated_data.append(populated_request)
    return generated_data

def generate_type3_parameter_variations(template_data):
    generated_data = []

    for email, team_id, dc, from_time, to_time in zip(template_data["variables"]["email"], template_data["variables"]["teamId"], template_data["variables"]["dc"], template_data["variables"]["from_time"], template_data["variables"]["to_time"]):
        # Replace placeholders with the actual values
        populated_request = {
            "input_text": template_data["input_text_template"].format(email=email, teamId=team_id, dc=dc, from_time=from_time, to_time=to_time),
            "api_request": {
                "endpoint": template_data["api_request"]["endpoint"],
                "method": template_data["api_request"]["method"],
                "headers": template_data["api_request"]["headers"],
                "data": {
                    "email": email,
                    "team_id": team_id,
                    "dc_id": dc,
                    "testType": "inboundvoice",
                    "start_time": from_time,
                    "end_time": to_time
                }
            }
        }
        generated_data.append(populated_request)
    return generated_data


def generate_trigger_parameter_variations(template_data):
    generated_data = []

    for duration, email, start_time in zip(template_data["variables"]["load_duration_hr"], template_data["variables"]["email"], template_data["variables"]["start_time"]):
        # Replace placeholders with the actual values
        populated_request = {
            "input_text": template_data["input_text_template"].format(load_duration_hr=duration, email=email, start_time=start_time),
            "api_request": {
                "endpoint": template_data["api_request"]["endpoint"],
                "method": template_data["api_request"]["method"],
                "headers": template_data["api_request"]["headers"],
                "data": {
                        "RunID": "run_id_to_replace_str",
                        "email": email,
                        "team_id": "teamid_replace_int",
                        "reservation_id": "reservation_id_to_replace_int",
                        "call_duration": 300,
                        "testType": "inboundvoice",
                        "dc_name": "loadus1",
                        "mailing_list": email,
                        "shifts": [
                            {
                            "call_profile": "calldummy.json",
                            "agent_profile": [
                                    "agendummy.csv"
                                ],
                            "duration": duration
                            }
                      ],
                      "load_duration": duration,
                      "fail_criterion": 85,
                      "start_time": start_time
                    }
                }
            }
        generated_data.append(populated_request)
    return generated_data

def generate_augmented_Dataset(variance: int):
    # Load training data from a JSON file
    with open('training_data.json', 'r') as f:
        training_data = json.load(f)

    # Augmented dataset to store results
    augmented_data = []

    # Generate variations for each entry in the training data
    for entry in training_data:
        if "type" in entry and entry['type'] == 'template1':
            # print("input_text_template", entry['input_text_template'])
            generated_entry_list = generate_type1_variations(entry)
            for generated_entry in generated_entry_list:
                input_text = generated_entry["input_text"]
                api_request = generated_entry["api_request"]
                # print("generating augmented text for :", input_text)
                # Generate variations
                variations = generate_variations(input_text, num_variations=variance)  # Adjust num_variations as needed

                # Create new entries with variations
                for variation in variations:
                    # print("adding variation to new dataset-", variation)
                    augmented_data.append({
                        "input_text": variation,
                        "api_request": api_request
                    })
        elif "type" in entry and entry['type'] == 'template2':
            generated_entry_list = generate_type2_parameter_variations(entry)
            for generated_entry in generated_entry_list:
                input_text = generated_entry["input_text"]
                api_request = generated_entry["api_request"]
                # print("generating augmented text for :", input_text)
                # Generate variations
                variations = generate_variations(input_text, num_variations=variance)  # Adjust num_variations as needed

                # Create new entries with variations
                for variation in variations:
                    print(variation)
                    print(api_request)
                    # print("adding variation to new dataset-", variation)
                    augmented_data.append({
                        "input_text": variation,
                        "api_request": api_request
                    })
        elif "type" in entry and entry['type'] == 'template3':
            generated_entry_list = generate_type3_parameter_variations(entry)
            for generated_entry in generated_entry_list:
                input_text = generated_entry["input_text"]
                api_request = generated_entry["api_request"]
                # print("generating augmented text for :", input_text)
                # Generate variations
                variations = generate_variations(input_text, num_variations=variance)  # Adjust num_variations as needed

                # Create new entries with variations
                for variation in variations:
                    # print("adding variation to new dataset-", variation)
                    augmented_data.append({
                        "input_text": variation,
                        "api_request": api_request
                    })
        elif "type" in entry and entry['type'] == 'template_trigger':
            generated_entry_list = generate_trigger_parameter_variations(entry)
            for generated_entry in generated_entry_list:
                input_text = generated_entry["input_text"]
                api_request = generated_entry["api_request"]
                # print("generating augmented text for :", input_text)
                # Generate variations
                variations = generate_variations(input_text, num_variations=variance)  # Adjust num_variations as needed

                # Create new entries with variations
                for variation in variations:
                    # print("adding variation to new dataset-", variation)
                    augmented_data.append({
                        "input_text": variation,
                        "api_request": api_request
                    })
        else:
            input_text = entry["input_text"]
            api_request = entry["api_request"]
            # print("generating augmented text for :", input_text)
            # Generate variations
            variations = generate_variations(input_text, num_variations=variance)  # Adjust num_variations as needed

            # Create new entries with variations
            for variation in variations:
                augmented_data.append({
                    "input_text": variation,
                    "api_request": api_request
                })

    # Print some of the augmented data as a preview
    # print(json.dumps(augmented_data, indent=2))

    # Save augmented data to a new JSON file
    with open('augmented_training_data.json', 'w') as f:
        json.dump(augmented_data, f, indent=2)


if __name__ == "__main__":
    generate_augmented_Dataset(variance=10)