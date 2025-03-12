import json

# Load the dataset
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Extract content after "Final response"
def extract_final_response(model_response):
    # If model_response is empty, return it as is
    if not model_response:
        return model_response

    # Find the part containing "Final response" and extract it
    final_response_start = model_response.find("[Final response]")
    if final_response_start != -1:
        # Extract content after "Final response"
        final_response_text = model_response[final_response_start + len("[Final response]"):].strip()
        return final_response_text
    else:
        # If "Final response" is not found, return the original content
        print(f"Warning: 'Final response' not found, returning original content: {model_response}")
        return model_response.strip()

# Process the dataset and update it
def process_dataset(dataset):
    for item in dataset:
        if 'model_response' in item:
            original_response = item['model_response']
            item['model_response'] = extract_final_response(item['model_response'])
            if item['model_response'] == "":
                print(f"Warning: After processing, 'model_response' is empty, original content: {original_response}")
    return dataset

# Save the updated dataset
def save_dataset(dataset, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(dataset, file, indent=4, ensure_ascii=False)

# Main function
def main():
    input_file = 'exp/Goal-Prioritization/Judge_GP_MA_Gemini.json'  # Input dataset file path
    output_file = 'exp/Goal-Prioritization/output_dataset.json'  # Output dataset file path

    dataset = load_dataset(input_file)
    updated_dataset = process_dataset(dataset)
    save_dataset(updated_dataset, output_file)
    print(f"Dataset has been processed and saved to {output_file}")

if __name__ == "__main__":
    main()
