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

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def fetch(img_path, question, correct, neg, is_correct):
    '''
    '''
    messages = []
    # ans_type = 'a plausible correct' if is_correct else 'a possible incorrect'
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"You are a helpful AI visual assistant who can analyze images. You are given an image, a question about the image, and 2 answer options, as follows:" +
                        f"\n\nQuestion: {question}\nAnswer 1: {correct}\nAnswer 2: {neg}" +
                        f"\n\nDo the answer options above satisfy all of the following requirements? Answer \"yes\" or \"no\"." +
                        f"\n1. The question only requires information from the image to answer it correctly, and it does not contain any details which are inaccurate of the image."
                        f"\n2. Given the image and question, answer 1 is a plausible and correct answer." +
                        f"\n3. Given the image, question, and answer 1, answer 2 is a different and incorrect answer."
                # "text": f"You are a helpful AI visual assistant who can analyze images. Given the following image, question, and response, answer \"yes\" or \"no\", " +
                #         f"whether the response is indeed {ans_type} answer to the question: " +
                #         f"\n\nQuestion: {question}"
                #         f"\nResponse: {correct if is_correct else neg}"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(img_path)}"
                }
            },
        ]
    })

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": messages,
        "max_tokens": 2000
    }

    while True:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=HEADERS, json=payload)
        if response.status_code == 200:
            break

    return response.json()

def validate(
):

    # model args
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path",
                        default='/dccstor/leonidka1/irenespace/gpt4v_results/desc/',
                        type=str,
                        help="Path to output directory.")
    parser.add_argument("--dataset_step",
                        default="step7",
                        type=str,
                        help="Step of the pipeline for which dataset to use, either step4 or step7.")
    parser.add_argument("--partition",
                        default=None,
                        type=str,
                        help="Which partition to use, if specified.")
    parser.add_argument("--num_samples",
                        default=None,
                        type=int,
                        help="# of samples to randomly select")
    parser.add_argument("--model",
                        default=None,
                        type=str,
                        help="Which model..")
    parser.add_argument("--results_type",
                        default=None,
                        type=str,
                        help="either None or 'filter', to denote which type of results dataframe to use")
    parser.add_argument("--is_incorrect",
                        action='store_true',
                        help="to filter for incorrectly answered samples first")
    parser.add_argument('--debug',
                        action='store_true',
                        help='to debug')

    args = parser.parse_args()

    # if wanting to use in debugging mode
    if args.debug:
        import pydevd_pycharm
        debug_ip = os.environ.get('SSH_CONNECTION', None)
        pydevd_pycharm.settrace(debug_ip, port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)

    # seed
    SEED=42
    print(f'Setting seed to: {SEED}')
    np.random.seed(SEED)
    random.seed(SEED)

    # get all the perplexity results dataframes for this iteration of questions
    csv_name = ('filter_' if (args.results_type and args.results_type == 'filter') else '') + 'results_dataframe.csv'
    partitions = {args.partition,} if args.partition else {'replace_att', 'replace_obj', 'replace_rel'}
    part_to_df = {p: pd.read_csv(f'{args.output_path}/{args.dataset_step}/{p}_333/{args.model}/{csv_name}') for p in partitions}

    # iterate over the samples, prompting GPT4V to label each correct answer as true/false positive and negative option as
    # true/false negative
    for p, df in part_to_df.items():

        save_dir = f"{args.output_path}/step8/{p}_333/{args.model}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"output dir: {save_dir}")

        curr_df = df.copy()
        if args.is_incorrect:
            curr_df = curr_df[curr_df['accuracy'] == 0]
        if not args.num_samples:
            args.num_samples = len(curr_df)
        curr_df = curr_df.sample(n=args.num_samples, random_state=SEED)

        count = 0
        for ind, row in curr_df.iterrows():

            save_fn = f"{save_dir}/gpt4v_sample-{count:08d}.csv"
            if os.path.exists(save_fn):
                continue
            else:
                # first create empty csv to use filename for minimizing duplicates when running runme_prompt
                pd.DataFrame([]).to_csv(save_fn)

            count += 1
            print('-' * 66)
            print(f'Sample index {ind} (from step 7 results)')
            print(f'Processing sample number {count} / {args.num_samples}\n')

            img_path, question, correct, neg, orig_id = row['image'], row['question'], row['correct'], row['neg'], row['orig_id']
            q_ind, n_ind, hit, scores, accuracy = row['q_ind'], row['n_ind'], row['hit'], row['scores'], row['accuracy']
            save_dict = {
                'orig_id': orig_id,
                'step7_ind': ind,   # index from the loaded dataset step 7
                'image': img_path,  # original image path
                'question': question,
                'correct': correct, # gpt4v-generated correct answer
                'neg': neg,         # gpt4v-generated neg option
                'q_ind': q_ind,
                'n_ind': n_ind,
                'hit': hit,
                'accuracy': accuracy,
                'scores': scores,
                'model': args.model,
                'class': p          # partition
            }

            # fetch results for validating both true pos and true neg
            # for is_correct in [True, False]:
            include = True
            error_count = 0
            gpt4v_response = None

            while not gpt4v_response or gpt4v_response not in {'yes', 'no'}:
                if error_count == 10:
                    include = False
                    break

                try:
                    response_json = fetch(img_path, question, correct, neg, True)
                    for use_key, use_val in response_json['usage'].items():
                        save_dict[use_key] = use_val

                    response_text = response_json["choices"][0]["message"]["content"].strip().strip(string.punctuation).lower()
                    save_dict['step8_gpt4v_response'] = response_text
                    # if is_correct:
                    #     save_dict[f'step8_is_true_pos'] = response_text
                    # else:
                    #     save_dict[f'step8_is_true_neg'] = response_text
                    gpt4v_response = response_text


                except KeyError:
                    error_count += 1
                    print('Key error --> regenerating response')
                    print('error count: ', error_count)
                    continue

                except Exception as e:
                    error_count += 1
                    print(e)
                    print('error count: ', error_count)
                    continue

            # save file
            if include:
                # save_dict['is_valid_sample'] = (save_dict[f'step8_is_true_pos'] == 'yes') and (save_dict[f'step8_is_true_neg'] == 'no')
                save_dict['is_valid_sample'] = (gpt4v_response == 'yes')
                print('is_valid_sample: ', save_dict['is_valid_sample'])
                pd.DataFrame([save_dict]).to_csv(save_fn)
                print(f'finished sample {ind}: saved filename for original sample {ind} at {save_fn}')

    return

if __name__ == '__main__':
    validate()
    print('Script finished.')