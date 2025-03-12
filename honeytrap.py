import os
# Set OpenAI API environment variables
os.environ["OPENAI_API_KEY"] = "***"
os.environ["OPENAI_BASE_URL"] = "***"

import json
import random
from argparse import ArgumentParser
from agentverse.agentverse import AgentVerse
from eval_helper.get_evaluation import get_evaluation
from agentverse.message import Message  # Import the Message class


parser = ArgumentParser()

# Parse command line arguments
parser.add_argument("--config", type=str, default="agentverse/tasks/honeytrap/config.yaml")
args = parser.parse_args()

print(args)
# Initialize AgentVerse
agentverse, args_data_path, args_output_dir = AgentVerse.from_task(args.config)

# Create output directory and save command line arguments
os.makedirs(args_output_dir, exist_ok=True)
with open(os.path.join(args_output_dir, "args.txt"), "w") as f:
    f.writelines(str(args))

# Read the dataset
with open(args_data_path) as f:
    data = json.load(f)

output = []

# Process each data instance
for num, ins in enumerate(data):
    print(f"================================instance {num}====================================")

    # Initialize the list of question fields
    turns = []
    
    # Select the question field based on the dataset
    if "Question" in ins:
        turns.append(ins["Question"])
    
    if "Rephrased_question" in ins:
        turns.append(ins["Rephrased_question"])

    if "Jailbreak_question" in ins:
        jailbreak_category = random.choice(list(ins["Jailbreak_question"].values()))
        jailbreak_question = random.choice(jailbreak_category)
        turns.append(jailbreak_question)
    

    target = ins["Target"]
# Process each round
    for turn in range(len(turns)):
        # Handle the response for each round
        # Dynamically choose the question field name
        if turns[turn] == ins["Question"]:
            question_field = "question"
        elif turns[turn] == ins["Rephrased_question"]:
            question_field = "Rephrased_question"
        else :
            question_field = "Jailbreak_question"
        print(f"Round {turn + 1}: {turns[turn]}")

        # Assign the input question to each agent
        for agent_id in range(len(agentverse.agents)):
            agentverse.agents[agent_id].source_text = turns[turn]
            agentverse.agents[agent_id].final_prompt = ""

        # Run AgentVerse
        agentverse.run() 

        # Print the response from each agent
        for agent_id in range(len(agentverse.agents)):
            print("Agent Name:", agentverse.agents[agent_id].name)
            if agentverse.agents[agent_id].memory.messages:
                print(f"Response from agent {agentverse.agents[agent_id].name}: {agentverse.agents[agent_id].memory.messages}")
            else:
                print(f"Response from agent {agentverse.agents[agent_id].name} is empty")
        
        # Get the evaluation result
        evaluation = get_evaluation(setting="every_agent", messages=agentverse.agents[0].memory.messages,
                                    agent_nums=len(agentverse.agents))

        output.append({ question_field: turns[turn], "target":ins["Target"], "evaluation": evaluation})

# Save the results to a file
os.makedirs(args_output_dir, exist_ok=True)

# Define the default serialization function
def default_serializer(obj):
    if isinstance(obj, Message):  # If the object is of type Message
        return {
            "content": obj.content,
            "sender": obj.sender,
            "receiver": list(obj.receiver) if isinstance(obj.receiver, set) else obj.receiver,  # Convert set to list
            "tool_response": obj.tool_response
        }
    raise TypeError(f"Type {obj.__class__.__name__} is not serializable")

# Output evaluation results to a JSON file
with open(os.path.join(args_output_dir, "attack_results.json"), "w") as f:
    json.dump(output, f, indent=4, default=default_serializer)
