import time
import json
import os
import sys
import random
import re
import string
import argparse
from functools import partial

# openai imports
import base64
import requests
from openai import OpenAI
from dotenv import load_dotenv

# import fire
import tqdm
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

from dec_vl_eval.src.datasets.benchmark_datasets import get_benchmark_dataset

load_dotenv()
API_KEY = os.getenv("OPENAI_KEY", None)
HEADERS = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {API_KEY}"
}

VLM_dict = {
    'llava-v1.6-7b': 'LLaVA 1.6-7b',
    'llava-v1.5-7b': 'LLaVA 1.5-7b',
    'instructblip_flan_t5': 'InstructBLIP Flan-T5',
    'instructblip_vicuna_7b': 'InstructBLIP Vicuna-7b'
}

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_describe_prompt(img_path):
    messages = []
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"You are a helpful AI visual assistant who can analyze images. Please describe this image in as much detail as possible. " +
                        "For all the details you are confident about, include everything you see, and be as specific as possible, such as describing objects, attributes, locations, lighting..."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(img_path)}"
                }
            },
        ]
    })

    return messages


def get_dim_prompt(img_path, dim):
    '''
    helper function for constructing the prompt to generate question, correct answer, and hard negatives altgether

    :return: string of the formatted prompt
    '''

    messages = []

    format_beg = 'Compositional reasoning defines the understanding of attributes, relations, and word order significance. ' \
            'A good vision-language model should be able to accurately answer composition reasoning questions about an image. ' \
            'Your task is to fool a vision-language model by generating challenging compositional reasoning questions about an image.' \
            '\n\nFor each question, include a correct answer and multiple negative options. Each negative option should be only subtly different ' \
            'from the correct answer but incorrect given the image and question. The goal is for a vision-language model ' \
            'to choose the negative option when answering the question in multiple choice format. Only include questions for which you are confident about the answer.'
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"{format_beg}"
            },
        ]
    })

    example_dict = {
        'first': {
            'img': '/dccstor/leonidka1/victor_space/data/coco/images/train2017/000000577819.jpg',
            'question': 'What is on the plate?',
            'correct': 'Knife',
            'neg': 'Spoon',
        },
        'second': {
            'img': '/dccstor/leonidka1/victor_space/data/coco/images/train2017/000000538965.jpg',
            'question': 'What color is the sweatshirt?',
            'correct': 'White',
            'neg': 'Gray',
        },
        'third': {
            'img': '/dccstor/leonidka1/victor_space/data/coco/images/train2017/000000212523.jpg',
            'question': 'How many women?',
            'correct': '10',
            'neg': '5',
        },
    }
    for k, val in example_dict.items():
        format_ex = f'Here is a {"" if k == "first" else k + " "}simple example of an image, a question, a correct answer, and a negative option which a vision-language model fails on:\nQuestion: {val["question"]}\nCorrect answer: {val["correct"]}\nNegative option: {val["neg"]}'
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{format_ex}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(val['img'])}"
                    }
                },
            ]
        })

    dim2prompt = {
        'spatial_relation': 'create complex questions about spatial relations between two objects. ' +
                            'To answer the questions, one should find the two mentioned objects and then find their relative spatial relation to answer the question.',
        'location': 'create complex questions about the location of objects in the image.',
        'reasoning': 'create complex questions beyond describing the scene. To answer such questions, one should first understand the visual content, ' +
                    "and then based on the background knowledge or reasoning, either explain why the things are happening that way, or provide guides and help to the user's request. " +
                    "Make the question challenging by not including the visual content details in the question so that the user needs to reason about that first."
    }
    format_dim = f'Now, it’s your turn. Using this image, {dim2prompt[dim]} Generate 3 sets of the following: ' \
                '\n- A challenging compositional reasoning question' \
                '\n- A correct answer\n- 5 hard negative options' \

    format_end = 'Format your response as a string in the format [{"q": <question>, "a": <correct answer>, "n1": <negative option 1>, "n2": <negative option 2>…}].'
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"{format_dim}\n\n{format_end}"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(img_path)}"
                }
            },
        ]
    })

    return messages

def fetch(gpt4v_type, messages, pipeline_step, responses=dict()):
    '''

    :param img_path:
    :param prompt:
    :return:
    '''
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": messages,
        "max_tokens": 2000
    }

    if pipeline_step >= 3:

        # gpt4v generated instance for step 1 (the description)
        payload['messages'].append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"{responses['step1_gpt4v']}"
                }
            ]
        })

        if gpt4v_type == 'dim':
            # vlms' responses + prompt again
            # response[f'{vlm}'] is in the format [<answer to q1>, <answer to q2>, ...]
            payload['messages'].append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Given the image and these questions, the following vision-language models generated their own respective answers, listed in order of the questions:\n" + \
                                '\n'.join([f"{format_name}: {responses[f'{code_name}']}" for code_name, format_name in VLM_dict.items()]) + \
                                '\n\nGenerate 3 new sets of question/answer/negatives which these vision-language models would find more challenging.'
                    }
                ]
            })

        elif gpt4v_type == 'desc':
            format_vlms = 'The following vision-language models generated their own respective descriptions for the provided image:\n' + \
                          '\n'.join([f"{format_name}: {responses[f'step2_{code_name}']}" for code_name, format_name in
                                     VLM_dict.items()])
            format_beg = 'Compositional reasoning defines the understanding of attributes, relations, and word order significance. ' \
                         'A good vision-language model should be able to accurately answer composition reasoning questions about an image. ' \
                         'Your task is to fool a vision-language model by generating challenging compositional reasoning questions about an image.'
            format_end = 'Given the description you generated and the descriptions these vision-language models generated, ' \
                         'generate 10 challenging compositional reasoning questions which these models would incorrectly answer. ' \
                         'Only create questions based on details captured in your description but lacking from the other vision-language models’ descriptions. ' \
                         'For each question, include the following:' \
                         '\n- A compositional reasoning question\n- A correct answer\n- 5 hard negative options' \
                         '\n\nEach negative option should differ only subtly ' \
                         'from the correct answer but still be clearly incorrect given the image and question. The goal is for a vision-language model ' \
                         'to choose the negative option over the positive option when asked to answer the question in binary multiple choice format. Only include questions you are confident in your answer for. ' \
                         'Format your response as a string in the format [{"q": <question>, "a": <correct answer>, "n1": <negative option 1>, "n2": <negative option 2>…}].'

            payload['messages'].append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{format_vlms}\n\n{format_beg}\n\n{format_end}"
                    }
                ]
            })

            if pipeline_step >= 6:
                # gpt4v generated instance for step 3 (the first iteration of generated questions)
                payload['messages'].append({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{responses['step3_gpt4v']}"
                        }
                    ]
                })

                payload['messages'].append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Given the image and these questions, the following vision-language models generated their own respective answers, listed in order of the questions:\n" + \
                                    '\n'.join(
                                        [f"{format_name}: {responses[f'step5_{code_name}']}" for code_name, format_name in
                                         VLM_dict.items()]) + \
                                    '\n\nGenerate 10 new sets of question/answer/negatives which these vision-language models would find even more challenging.'
                        }
                    ]
                })

        else:
            print(f'{gpt4v_type} not yet implemented for gpt4v_type.')

    while True:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=HEADERS, json=payload)
        if response.status_code == 200:
            break

    return response.json()

def prompt(
    output_path: str,
):

    # model args
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument("--eval_dataset",
                        default='SUGAR',
                        type=str,
                        help="Which dataset to use for evaluation.")
    parser.add_argument("--partition",
                        default=None,
                        type=str,
                        help="Which partition of the dataset, if any.")
    parser.add_argument("--dim",
                        default=None,
                        type=str,
                        help="Which dimension to use for the prompt.")
    parser.add_argument("--num_samples",
                        default=None,
                        type=int,
                        help="Optional number of samples to limit from the dataset.")
    parser.add_argument("--select_by",
                        default='loss_score',
                        type=str,
                        help="Which evaluation method to use.")

    parser.add_argument("--output_path",
                        default=output_path,
                        type=str,
                        help="Path to output directory.")
    parser.add_argument("--job_name",
                        default="1",
                        type=str,
                        help="Job name that is used in output directory path.")
    parser.add_argument("--pipeline_step",
                        default=1,
                        type=int,
                        help="Step of the pipeline, either 1 or 3.")
    parser.add_argument("--gpt4v_type",
                        default="dim", # either dim or desc
                        type=str,
                        help="Which type of generation to use for step 1 prompting.")
    parser.add_argument('--debug',
                        action='store_true',
                        help='to debug')

    args = parser.parse_args()

    # if wanting to use in debugging mode
    if args.debug:
        import pydevd_pycharm
        debug_ip = os.environ.get('SSH_CONNECTION', None)
        pydevd_pycharm.settrace(debug_ip, port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)

    # check that output path dir exists
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        # print(f"output dir: {args.output_path}")

    # get requested dataset by using helper function from benchmarks
    data_dict, options = get_benchmark_dataset(args=args)

    if args.pipeline_step >= 3:
        # TODO: update this with the correct path
        vlm_dfs = {
            f'step2_{vlm}': pd.read_csv(
                f'/dccstor/leonidka1/irenespace/gpt4v_results/desc/step2/{args.job_name}/{vlm}/results_dataframe.csv'
            ) for vlm in VLM_dict.keys()
        }
        if args.pipeline_step > 3:
            vlm_dfs.update({
                f'step5_{vlm}': pd.read_csv(
                    f'/dccstor/leonidka1/irenespace/gpt4v_results/desc/step5/{args.job_name}/{vlm}/results_dataframe.csv'
                ) for vlm in VLM_dict.keys()
            })

    # seed
    # max_int_32bit = 2 ** 32 - 1
    # SEED = int(round(time.time() * 1000)) % max_int_32bit
    SEED=42
    # print(f'Setting seed to: {SEED}')
    np.random.seed(SEED)
    random.seed(SEED)

    for key, samples in data_dict.items():

        all_samples = random.sample(samples, k=len(samples))

        if args.eval_dataset == 'SUGAR' and args.partition and key != args.partition:
            continue

        # set of original dataset indices to include
        indices = set(i for i in range(len(all_samples)))
        if args.pipeline_step == 3:
            indices = set(vlm_dfs['step2_llava-v1.6-7b']['orig_id'])
        elif args.pipeline_step == 6:
            indices = set(vlm_dfs['step5_llava-v1.6-7b']['orig_id'])

        for ind, sample in enumerate(all_samples): # this ind and sample is based on sugarcrepe's original ordering/dataset

            if ind not in indices:
                continue

            if args.num_samples and ind == args.num_samples:
                break

            save_fn = f"{args.output_path}/gpt4v_{key}_sample-{ind:08d}.csv"
            if os.path.exists(save_fn):
                continue
            else:
                # first create empty csv to use filename for minimizing duplicates when running runme_prompt
                pd.DataFrame([]).to_csv(save_fn)

            # print('-' * 66)
            # print(f'Seeded sample index {ind}\n')
            # # print(f'Original sample index {sample[-1]}\n')

            pos = sample[2] if sample[2] else None
            # print('original positive: ', pos)

            neg = [text for text in sample[-2] if text != pos][0] if (sample[-2] is not None) else None
            # print('original negative: ', neg)

            img_path = sample[0]
            save_dict = {
                'orig_ind': ind,  # original index from the loaded dataset
                'image_path': img_path,  # original image path
                'orig_pos': pos,  # original positive
                'orig_neg': neg,  # original negative
            }

            gen_dict = {'q': None}
            if args.gpt4v_type == 'dim' or args.pipeline_step >= 3:
                gen_dict.update({'a': None})
                gen_dict.update({k: None for k in [f'n{i}' for i in range(1, 6)]})

            include = True
            error_count = 0
            while None in gen_dict.values():
                if error_count == 10:
                    include = False
                    break

                try:
                    if args.gpt4v_type == 'dim':
                        prompt = get_dim_prompt(img_path, args.dim)
                    elif args.gpt4v_type == 'desc':
                        prompt = get_describe_prompt(img_path)
                    else:
                        print('This generation type for step 1 has not been implemented.')

                    vlm_responses = dict()
                    if args.pipeline_step >= 3:
                        gen_steps = ['step2'] if args.pipeline_step == 3 else ['step2', 'step5']

                        # to use for ICL examples and continuing the conversation
                        for step in gen_steps:
                            for vlm in VLM_dict.keys():
                                curr_df = vlm_dfs[f'{step}_{vlm}']
                                row = curr_df[curr_df['orig_id'] == ind]

                                if args.gpt4v_type == 'dim':
                                    for q_ind in range(1,4):
                                        if f'step1_q{q_ind}_q' not in save_dict.keys():
                                            save_dict[f'step1_q{q_ind}_q'] = row[f'q{q_ind}_q'].values[0]

                                if step == 'step2' and 'step1_gpt4v_response' not in vlm_responses.keys():
                                    vlm_responses[f'step1_gpt4v'] = row['gpt4v_response'].values[0]
                                    save_dict['step1_gpt4v_response'] = row['gpt4v_response'].values[0]

                                if step == 'step5' and 'step3_gpt4v_response' not in vlm_responses.keys():
                                    vlm_responses[f'step3_gpt4v'] = row['gpt4v_response'].values[0]
                                    save_dict['step3_gpt4v_response'] = row['gpt4v_response'].values[0]

                                vlm_responses[f'{step}_{vlm}'] = row['gen_text'].values[0]
                                save_dict[f'{step}_{vlm}_response'] = row['gen_text'].values[0]

                    response_json = fetch(args.gpt4v_type, prompt, args.pipeline_step, vlm_responses)
                    for use_key, use_val in response_json['usage'].items():
                        save_dict[use_key] = use_val

                    response_text = response_json["choices"][0]["message"]["content"].strip().strip('```json').strip()
                    save_dict[f'step{args.pipeline_step}_gpt4v_response'] = response_text

                    if args.gpt4v_type == 'dim' or args.pipeline_step >= 3:
                        response_list = json.loads(response_text)
                        for q_ind in range(10):
                            response_dict = response_list[q_ind]
                            assert gen_dict.keys() == response_dict.keys()

                            for k, val in response_dict.items():
                                gen_dict[k] = val
                                save_dict[f'step{args.pipeline_step}_q{q_ind + 1}_{k}'] = str(val) # 1-indexed to be consistent with indexing for negs
                    else:
                        gen_dict['q'] = response_text

                except json.decoder.JSONDecodeError:
                    error_count += 1
                    # print('Encountered json error in parsing model responses --> regenerating response')
                    # print('response text: ', response_text)
                    # print('error count: ', error_count)
                    continue

                except KeyError:
                    error_count += 1
                    # print('Key error --> regenerating response')
                    # print('response text: ', response_text)
                    # print('error count: ', error_count)
                    continue

                except Exception as e:
                    error_count += 1
                    # print(e)
                    # print('error count: ', error_count)
                    continue

            # save file
            if include:
                pd.DataFrame([save_dict]).to_csv(save_fn)
                # print(f'finished seeded sample {ind}: saved filename for original sample {ind} at {save_fn}')

    return

if __name__ == '__main__':
    # output directory
    job = time.strftime("%Y%m%d_%H%M")
    OUTPUT_DIR = f"/dccstor/leonidka1/irenespace/llm_results_updated/negs/{job}"
    prompt(output_path=OUTPUT_DIR)
    print('Script finished.')