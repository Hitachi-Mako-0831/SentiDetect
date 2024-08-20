
"""Using LLaMa to rewrite for cheap detection """
import os
import numpy as np
import torch
import json

from transformers import AutoTokenizer,AutoModelForCausalLM
model_path = "./bigmodels/llama2_chat/converted_weights_llama_chat7b"
modeltype = 'llama2_7b_chat'

def tokenize_and_normalize(sentence):
    # Tokenization and normalization
    return [word.lower().strip() for word in sentence.split()]


tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")


import os
import openai
import json

debug = False
def GPT_self_prompt(prompt_str, content_to_be_detected):
    prompts = f"{prompt_str}: \"{content_to_be_detected}\""
    model_inputs = tokenizer(prompts, return_tensors="pt").to("cuda:0")
    model_inputs.pop("token_type_ids", None)

    output = model.generate(**model_inputs, max_new_tokens=len(tokenize_and_normalize(prompts)))

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    print('length', len(tokenize_and_normalize(prompts)), len(prompts))
    print(decoded_output)

    return decoded_output

prompt_list = []

with open(f'arXiv_human.json', 'r') as file:
    human = json.load(file)

with open(f'arxiv_GPT_concise.json', 'r') as file:
    GPT = json.load(file)


def rewrite_json(input_json, prompt_list, human=False):
    all_data = []
    for cc, data in enumerate(input_json):
        tmp_dict ={}
        
        tmp_dict['input'] = data['abs']

        for ep in prompt_list:
            tmp_dict[ep] = GPT_self_prompt(ep, tmp_dict['input'])
        
        all_data.append(tmp_dict)

        if debug:
            break
    return all_data

human_rewrite = rewrite_json(human, prompt_list, True)
with open(f'.json', 'w') as file:
    json.dump(human_rewrite, file, indent=4)

GPT_rewrite = rewrite_json(GPT, prompt_list)
with open(f'.json', 'w') as file:
    json.dump(GPT_rewrite, file, indent=4)





