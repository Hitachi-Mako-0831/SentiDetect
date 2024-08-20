import os
import openai
import json

openai.api_key='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'

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openai_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

debug=False
max_tokens = 300
# modeltype = "text-davinci-002"
modeltype = 'ada'


def GPT_self_prompt(prompt_str, content_to_be_detected):

    # import pdb; pdb.set_trace()
    response = openai_backoff(
                    model=modeltype,
                    prompt= f"{prompt_str}: \"{content_to_be_detected}\"",
                    temperature=0,
                    max_tokens = max_tokens
                )
    # spit_out = response["choices"][0]["message"]["content"].strip()
    spit_out = response["choices"][0]["text"].strip()

    print(spit_out)
    return spit_out

prompt_list = []

with open(f'arXiv_human.json', 'r') as file:
    human = json.load(file)

with open(f'arxiv_GPT.json', 'r') as file:
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
with open(f'{modeltype}_rewrite_arxiv_human.json', 'w') as file:
    json.dump(human_rewrite, file, indent=4)

GPT_rewrite = rewrite_json(GPT, prompt_list)
with open(f'{modeltype}_rewrite_arxiv_GPT.json', 'w') as file:
    json.dump(GPT_rewrite, file, indent=4)



