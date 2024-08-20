import os
import openai
import json

openai.api_key=

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openai_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

debug=False


def GPT_self_prompt(prompt_str, content_to_be_detected):


    response = openai_backoff(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "user",
                            "content": f"{prompt_str}: \"{content_to_be_detected}\"",
                        }
                    ],
                )
    spit_out = response["choices"][0]["message"]["content"].strip()
    print(spit_out)
    return spit_out

prompt_list = []


filelist = [('.json', '.json'),
            ('.json', '.json')
            ]

filelist = [('.json', '.json')
            ]
filelist = [('.json', '.json')
            ]


for inputname, outputname in filelist:
    with open(inputname, 'r') as file:
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

    GPT_rewrite = rewrite_json(GPT, prompt_list)
    with open(outputname, 'w') as file:
        json.dump(GPT_rewrite, file, indent=4)




