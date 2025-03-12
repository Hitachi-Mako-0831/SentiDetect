import requests
import json
from judge_template import JudgeTemplate

# Set the OpenAI API URL and key
url = ""
api_key = "" 

# Load the dataset file
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Send request and process target and model response
def send_request_and_process(goal, model_response):
    # Use the JudgeTemplate class to generate a filled template
    judge = JudgeTemplate(goal, model_response)
    filled_judge_template = judge.generate_template()

    # Set the request headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Define the request body (content), passing the filled judge template as the content
    data = {
        "max_tokens": 1200,
        "model": "gpt-3.5-turbo",  
        "temperature": 0.8,
        "top_p": 1,
        "presence_penalty": 1,
        "messages": [
            {
                "role": "system",
                "content": "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible."
            },
            {
                "role": "user",
                "content": filled_judge_template  # Pass the filled judge template as the content
            }
        ]
    }

    # Send a POST request to the API
    response = requests.post(url, headers=headers, data=json.dumps(data).encode('utf-8'))

    # Get the response content
    result = response.content.decode("utf-8")

    # Parse the returned JSON response (if needed)
    response_data = json.loads(result)

    # Extract the model's reply content
    if "choices" in response_data and len(response_data["choices"]) > 0:
        # If "choices" exists and is not empty, extract the first choice's content
        chat_response = response_data["choices"][0]["message"]["content"]
    else:
        # If "choices" does not exist or is empty, it means there is no valid response
        print("No valid response returned")
        chat_response = ""

    # Separate `reason` and `score` parts
    # The reason and score sometimes include "#score:" as a delimiter
    reason_start = chat_response.find("#thereason:")
    score_start = chat_response.find("#thescore:")

    if reason_start != -1 and score_start != -1:
        reason = chat_response[reason_start + len("#thereason:"):score_start].strip()
        score = chat_response[score_start + len("#thescore:"):].strip()
    else:
        reason = "No reason provided."
        score = "No score provided."

    return reason, score

# Process all targets and responses in the dataset
def process_all_targets_and_responses(dataset):
    all_results = []
    
    for idx, entry in enumerate(dataset, 1):
        target = entry['goal']
        model_response = entry['chat_response']
        
        # Dynamically retrieve the question fields
        question = entry.get('question', None)
        rephrased_question = entry.get('Rephrased_question', None)
        jailbreak_question = entry.get('Jailbreak_question', None)
        
        # Dynamically decide which field to use based on the available one
        if question:
            selected_question_field = 'question'
            selected_question = question
        elif rephrased_question:
            selected_question_field = 'Rephrased_question'
            selected_question = rephrased_question
        elif jailbreak_question:
            selected_question_field = 'Jailbreak_question'
            selected_question = jailbreak_question
        else:
            continue  # Skip if no question fields are available
        
        if model_response:
            print(f"Processing the {idx}th entry...")

            reason, score = send_request_and_process(target, model_response)
            
            # Save the selected question field as the field name, keeping other information unchanged
            all_results.append({
                selected_question_field: selected_question,  # Save the question using the field name as the key
                "goal": target,
                "model_response": model_response,
                "reason": reason,
                "score": score
            })

    return all_results

# Save the results to a file
def save_results_to_file(results, output_file="exp/Our_method/Judge_ours.json"):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Data has been saved to {output_file}")

# Main program entry point
dataset = load_dataset('exp/Our_method/result_ours.json')  # Replace with the actual file path
results = process_all_targets_and_responses(dataset)
save_results_to_file(results)  # Save the results to a file
