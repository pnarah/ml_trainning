import json

def generate_type1_variations(template_data):
    generated_data = []
    # Loop over each template entry
    api_name_template = template_data["api_name_template"]
    description = template_data["description"]
    endpoint_template = template_data["endpoint_template"]
    method = template_data["method"]
    variables = template_data["variables"]

    key, values = list(variables.items())[0]
    # Loop over all possible combinations of variables
    for value in values:
        # Substitute the single variable in the template
        api_name = api_name_template.format(**{key: value})
        description = description.format(**{key: value})
        endpoint = endpoint_template.format(**{key: value})

        form_data = {
            "api_name": api_name,
            "description" : description,
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
    generated_data = []

    for email, team_id in zip(template_data["variables"]["email"], template_data["variables"]["teamId"]):
        # Replace placeholders with the actual values
        populated_request = {
            "api_name": template_data["api_name_template"].format(email=email, teamId=team_id),
            "description": template_data["description"].format(email=email, teamId=team_id),
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
        generated_data.append(populated_request)
    return generated_data


def generate_augmented_Dataset():
    # Load training data from a JSON file
    with open('training_data_rag.json', 'r') as f:
        training_data = json.load(f)

    # Augmented dataset to store results
    augmented_data = []

    # Generate variations for each entry in the training data
    for entry in training_data:
        if "type" in entry and entry['type'] == 'template1':
            generated_entry_list = generate_type1_variations(entry)
            for generated_entry in generated_entry_list:
                augmented_data.append({
                    "api_name": generated_entry["api_name"],
                    "description": generated_entry["description"],
                    "api_request": generated_entry["api_request"]
                })
        elif "type" in entry and entry['type'] == 'template2':
            generated_entry_list = generate_type2_parameter_variations(entry)
            for generated_entry in generated_entry_list:
                augmented_data.append({
                    "api_name": generated_entry["api_name"],
                    "description": generated_entry["description"],
                    "api_request": generated_entry["api_request"]
                })
        else:
            augmented_data.append({
                "api_name": entry["api_name"],
                "description": entry["description"],
                "api_request": entry["api_request"]
            })

    # Print some of the augmented data as a preview
    # print(json.dumps(augmented_data, indent=2))

    # Save augmented data to a new JSON file
    with open('augmented_training_data_rag.json', 'w') as f:
        json.dump(augmented_data, f, indent=2)


if __name__ == "__main__":
    generate_augmented_Dataset()