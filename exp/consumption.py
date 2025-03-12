import tiktoken
import json
import math

# Select model:
model = "gpt-3.5-turbo"  

# Get the tokenizer for the corresponding model
encoding = tiktoken.get_encoding("cl100k_base") if model == "gpt-3.5-turbo" else tiktoken.get_encoding("p50k_base")

# Read the dataset JSON file
with open('XX', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# Initialize the result list
results = []

# Initialize the total token consumption for each question type and response
total_question_tokens = 0
total_rephrased_question_tokens = 0
total_jailbreak_question_tokens = 0

total_question_response_tokens = 0
total_rephrased_question_response_tokens = 0
total_jailbreak_question_response_tokens = 0

total_tokens = 0  # Used to calculate the total token count

# Initialize the counts for each question type
question_count = 0
rephrased_question_count = 0
jailbreak_question_count = 0

# Iterate over the dataset and calculate token counts for each entry
for idx, entry in enumerate(dataset):
    # Get the question
    question = entry.get("question", "")
    rephrased_question = entry.get("Rephrased_question", "")
    jailbreak_question = entry.get("Jailbreak_question", "")
    
    # Get the corresponding response
    model_response = entry.get("model_response", "")
    
    # Calculate the token count for each question
    question_tokens = encoding.encode(question)
    rephrased_question_tokens = encoding.encode(rephrased_question)
    jailbreak_question_tokens = encoding.encode(jailbreak_question)
    
    # Calculate the token count for the response
    model_response_tokens = encoding.encode(model_response)
    
    # Default assignment as empty lists, indicating no response
    question_response_tokens = []
    rephrased_question_response_tokens = []
    jailbreak_question_response_tokens = []

    # Determine the response type based on the index of the question
    if idx % 3 == 0:  # Corresponds to `question`
        question_response_tokens = model_response_tokens
        question_count += 1
    elif idx % 3 == 1:  # Corresponds to `Rephrased_question`
        rephrased_question_response_tokens = model_response_tokens
        rephrased_question_count += 1
    else:  # Corresponds to `Jailbreak_question`
        jailbreak_question_response_tokens = model_response_tokens
        jailbreak_question_count += 1
    
    # Calculate the token count for each question type
    question_token_count = len(question_tokens)
    rephrased_question_token_count = len(rephrased_question_tokens)
    jailbreak_question_token_count = len(jailbreak_question_tokens)
    
    # Calculate the token count for each response type
    question_response_token_count = len(question_response_tokens)
    rephrased_question_response_token_count = len(rephrased_question_response_tokens)
    jailbreak_question_response_token_count = len(jailbreak_question_response_tokens)

    # Accumulate the token count for each question type and response
    total_question_tokens += question_token_count
    total_rephrased_question_tokens += rephrased_question_token_count
    total_jailbreak_question_tokens += jailbreak_question_token_count

    total_question_response_tokens += question_response_token_count
    total_rephrased_question_response_tokens += rephrased_question_response_token_count
    total_jailbreak_question_response_tokens += jailbreak_question_response_token_count

    # Calculate the total token consumption for each data entry
    total_token_count = (question_token_count + question_response_token_count +
                         rephrased_question_token_count + rephrased_question_response_token_count +
                         jailbreak_question_token_count + jailbreak_question_response_token_count)

    # Save the result to the list
    results.append({
        "question_token_count": question_token_count,
        "rephrased_question_token_count": rephrased_question_token_count,
        "jailbreak_question_token_count": jailbreak_question_token_count,
        "question_response_token_count": question_response_token_count,
        "rephrased_question_response_token_count": rephrased_question_response_token_count,
        "jailbreak_question_response_token_count": jailbreak_question_response_token_count,
        "total_token_count": total_token_count
    })

    # Accumulate the total token count
    total_tokens += total_token_count

# Calculate the average token consumption for each question type
average_question_tokens = total_question_tokens / question_count if question_count > 0 else 0
average_rephrased_question_tokens = total_rephrased_question_tokens / rephrased_question_count if rephrased_question_count > 0 else 0
average_jailbreak_question_tokens = total_jailbreak_question_tokens / jailbreak_question_count if jailbreak_question_count > 0 else 0

# Calculate the average token consumption for each response type
average_question_response_tokens = total_question_response_tokens / question_count if question_count > 0 else 0
average_rephrased_question_response_tokens = total_rephrased_question_response_tokens / rephrased_question_count if rephrased_question_count > 0 else 0
average_jailbreak_question_response_tokens = total_jailbreak_question_response_tokens / jailbreak_question_count if jailbreak_question_count > 0 else 0

# Calculate the average token consumption per group
average_tokens_per_group = total_tokens / len(dataset) if len(dataset) > 0 else 0


# Save the results to a new file
output_data = {
    "results": results,
    "average_question_tokens": average_question_tokens,
    "average_rephrased_question_tokens": average_rephrased_question_tokens,
    "average_jailbreak_question_tokens": average_jailbreak_question_tokens,
    "average_question_response_tokens": average_question_response_tokens,
    "average_rephrased_question_response_tokens": average_rephrased_question_response_tokens,
    "average_jailbreak_question_response_tokens": average_jailbreak_question_response_tokens,
    "average_tokens_per_group": average_tokens_per_group,
   
}

# Save the results to a JSON file
with open('', 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=4)

# Output the results
print(f"Average tokens for question: {average_question_tokens:.2f}")
print(f"Average tokens for rephrased question: {average_rephrased_question_tokens:.2f}")
print(f"Average tokens for jailbreak question: {average_jailbreak_question_tokens:.2f}")
print(f"Average tokens for question response: {average_question_response_tokens:.2f}")
print(f"Average tokens for rephrased question response: {average_rephrased_question_response_tokens:.2f}")
print(f"Average tokens for jailbreak question response: {average_jailbreak_question_response_tokens:.2f}")
print(f"Average tokens per group: {average_tokens_per_group:.2f}")
