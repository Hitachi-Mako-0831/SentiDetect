import os
import json
from typing import List
from agentverse.message import Message
import re

def get_evaluation(setting: str = None, messages: List[Message] = None, agent_nums: int = None) -> List[dict]:
    evaluation = []
    
    # Ensure we don't exceed the range of the messages list
    for i in range(agent_nums):
        if i < len(messages):  # Check if the index is valid
            agent_response = messages[i]
            agent_name = agent_response.sender
            # Use regular expression to split the content, matching one or more newline characters as separators
            content = re.split(r'\n{1,2}', agent_response.content, maxsplit=2)  # Split into at most two parts
            # If the agent is the Forensic Agent, directly store the response content
            if "Forensic" in agent_name:  # Assuming the forensic agent's name is "ForensicAgent"
                evaluation.append({
                    "agent_name": agent_name,
                    "response": agent_response.content,
                    "reason": ""  # Forensic Agent does not need splitting, leave reason empty
                })
            else:
                # Create the evaluation dictionary
                evaluation_dict = {
                    "agent_name": agent_response.sender,
                    "response": content[0],  # Take the first part as response
                    "reason": content[1] if len(content) > 1 else ""  # Take the second part as reason, or empty if not present
                }
                evaluation.append(evaluation_dict)
        else:
            # If the message is empty or out of range, handle it as None or other values
            evaluation.append({
                "agent_name": "Unknown",
                "response": "No valid response",
                "reason": ""
            })
    
    return evaluation
