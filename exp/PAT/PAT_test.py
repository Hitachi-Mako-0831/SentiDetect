import requests
import json
from PAT import Template
import random

# Set the OpenAI API URL and API key
url = ""
api_key = ""  

# Load the dataset file
def load_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Send request and process the target and model response
def send_request_and_process(query):
    judge = Template(query)
    filled_judge_template = judge.generate_template()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Define the data body (request content), using the filled judge template as content
    data = {
        "max_tokens": 1200,
        "model": "gpt-4",  
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
                "content": filled_judge_template  # Pass the filled judge template as content
            }
        ]
    }

    # Send a POST request to the API
    response = requests.post(url, headers=headers, data=json.dumps(data).encode('utf-8'))

    # Get the response content
    result = response.content.decode("utf-8")

    # Parse the returned JSON response (if necessary)
    response_data = json.loads(result)

    # Extract the model's reply content
    if "choices" in response_data and len(response_data["choices"]) > 0:
        chat_response = response_data["choices"][0]["message"]["content"]
    else:
        print("No valid response returned")
        chat_response = ""
    return chat_response

# Process each target and model response in the dataset
def process_all_targets_and_responses(dataset):
    all_results = []
    
    for num, ins in enumerate(dataset):
        print(f"================================instance {num}====================================")

        # Initialize the list for question fields
        turns = []
        
        # Select question fields based on the dataset
        if "Question" in ins:
            turns.append(ins["Question"])
        
        if "Rephrased_question" in ins:
            turns.append(ins["Rephrased_question"])

        if "Jailbreak_question" in ins:
            jailbreak_category = random.choice(list(ins["Jailbreak_question"].values()))
            jailbreak_question = random.choice(jailbreak_category)
            turns.append(jailbreak_question)
        
        # Target field
        target = ins.get("Target", "")

        # Process each round of responses
        for turn in turns:
            # Dynamically select the field name for the question
            if turn == ins.get("Question"):
                question_field = "question"
            elif turn == ins.get("Rephrased_question"):
                question_field = "Rephrased_question"
            elif turn == jailbreak_question:
                question_field = "Jailbreak_question"
            else:
                question_field = "unknown"

            # Get the model's response
            chat_response = send_request_and_process(turn)

            # Save the result
            all_results.append({
                question_field: turn,  # Save the question with the selected field name
                "goal": target,
                "chat_response": chat_response
            })

    return all_results

# Save the results to a file
def save_results_to_file(results, output_file="exp/PAT/result_PAT.json"):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Data has been saved to {output_file}")

# Main program entry
dataset = load_dataset('dataset/jailbreak_questions_MA.json')  # Replace with the actual file path
results = process_all_targets_and_responses(dataset)
save_results_to_file(results)  # Save the results to a file
