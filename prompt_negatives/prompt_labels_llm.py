import time
import json
import os
import sys
import random
import re
import string
import argparse
from functools import partial

# import fire
import tqdm
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

from dotenv import load_dotenv
from genai import Credentials, Client
# from genai.model import Model
from genai.schema import TextGenerationParameters, TextGenerationReturnOptions, DecodingMethod
# from genai.schemas.generate_params import LengthPenalty

from dec_vl_eval.src.datasets.benchmark_datasets import get_benchmark_dataset

load_dotenv()
API_KEY = os.getenv("GENAI_KEY", None)
API_URL = os.getenv("GENAI_API", None)

def prepare_prompt(path, filter_dataset):
    '''
    helper function for constructing the prompt for labeling

    :return: string of the formatted prompt
    '''

    def add_example(prompt, question, s1, s2, label):
        return prompt + f'\n\nQuestion: {question}\nS1: "{s1.strip().strip(string.punctuation)}"' \
                        f'\nS2: "{s2.strip().strip(string.punctuation)}"\nResult: {label}'


    samples = pd.read_csv(f'{path}/verif_50.csv')
    num = min(len(samples[samples['label'] == 'good']), len(samples[samples['label'] == 'bad']))
    good = samples[samples['label'] == 'good'].sample(n=num, random_state=42)
    bad = samples[samples['label'] == 'bad'].sample(n=num, random_state=42)
    merged = pd.concat([good, bad], ignore_index=True).sample(frac=1, random_state=42)

    prompt = 'You are given a question about an image and two possible answers, S1 and S2, to compare. ' \
            'Your task is to choose a label "good" or "bad". You should choose good only if the two answers are ' \
            'different in meaning and both plausible answers to the given question.\n\nHere are some examples:'

    for i in range(len(merged)):
        curr = merged.iloc[i]
        if filter_dataset == 'llava-mix':
            prompt = add_example(prompt, curr['prompt'], curr['correct_option'], curr['new_neg'], curr['label'])
        elif filter_dataset == 'SAMPLE_NEG_UPDATED':
            prompt = add_example(prompt, curr['prompt_eval'], curr['correct_option_eval'], curr['new_neg_verif'], curr['label'])

    # # 2 good
    # for i in range(2):
    #     curr = good.iloc[i]
    #     prompt = add_example(prompt, curr['prompt'], curr['correct_option'], curr['new_neg'], curr['label'])
    #
    # # 1 bad, then 1 good
    # for item in [good, bad]:
    #     for i in range(1):
    #         curr = item.iloc[i]
    #         prompt = add_example(prompt, curr['prompt'], curr['correct_option'], curr['new_neg'], curr['label'])
    #
    # # 2 bad
    # for i in range(2):
    #     curr = bad.iloc[i]
    #     prompt = add_example(prompt, curr['prompt'], curr['correct_option'], curr['new_neg'], curr['label'])

    return prompt, set(samples['original_id'].unique())

def label(
    output_path: str
):

    # model args
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",
                        default='ibm-mistralai/mixtral-8x7b-instruct-v01-q',
                        type=str,
                        help="Which language model to use.")
    parser.add_argument("--eval_vlm",
                        default='llava-v1.6-7b',
                        type=str,
                        help="Which model was used for evaluation.")
    parser.add_argument("--prompt_model",
                        default='llm',
                        type=str,
                        help="Which model type for prompting was used.")

    # dataset args
    parser.add_argument("--filter_dataset",
                        default='llava-mix',
                        type=str,
                        help="Which dataset was used for filtering.")
    parser.add_argument("--partition",
                        default=None,
                        type=str,
                        help="Which partition of the dataset, if any.")
    parser.add_argument("--num_samples",
                        default=None,
                        type=int,
                        help="Optional number of samples to limit from the dataset.")

    parser.add_argument("--output_path",
                        default=output_path,
                        type=str,
                        help="Path to output directory.")
    parser.add_argument('--debug',
                        action='store_true',
                        help='to debug')

    # generation args
    parser.add_argument("--genai_key",
                        default=API_KEY,
                        type=str,
                        help="Which BAM api key to use.")
    parser.add_argument("--generate_max_len",
                        default=5,
                        type=int,
                        help="Parameter for max number of token length.")
    parser.add_argument("--temperature",
                        default=0,
                        type=float,
                        help="Parameter for temperature.")
    parser.add_argument("--top_p",
                        default=1,
                        type=int,
                        help="Parameter for top_p.")
    parser.add_argument("--top_k",
                        default=100,
                        type=int,
                        help="Parameter for top_k.")

    args = parser.parse_args()

    # if wanting to use in debugging mode
    if args.debug:
        import pydevd_pycharm
        debug_ip = os.environ.get('SSH_CONNECTION', None)
        pydevd_pycharm.settrace(debug_ip, port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)

    # check that output path dir exists
    label_path = os.path.join(args.output_path, 'labels')
    if not os.path.exists(label_path):
        os.makedirs(label_path)
        print(f"output label dir: {label_path}")

    creds = Credentials(args.genai_key, api_endpoint=API_URL)
    params = TextGenerationParameters(
        decoding_method=DecodingMethod.SAMPLE,
        max_new_tokens=args.generate_max_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        random_seed=42
        # repetition_penalty=1.0,
        # length_penalty=LengthPenalty(decay_factor=1.3)
        # truncate_input_tokens=max_seq_len
    )

    client = Client(credentials=creds)


    # # seed
    # max_int_32bit = 2 ** 32 - 1
    # SEED = int(round(time.time() * 1000)) % max_int_32bit
    # print(f'Setting seed to: {SEED}')
    # np.random.seed(SEED)
    # random.seed(SEED)

    # # prepare prompt and set of ids to skip
    # prompt, skip_ids = prepare_prompt(args.output_path, args.filter_dataset)

    root = f"/dccstor/leonidka1/irenespace/{args.prompt_model}_results_updated/eval/"  # change the ending subfolder
    if args.filter_dataset == 'SAMPLE_NEG_UPDATED':
        results_path = f"{root}/ed_MODEL_SPEC_UPDATED_LLM_fd_SAMPLE_NEG_UPDATED_ev_{args.eval_vlm}_dp_{args.partition}_wn_new_neg_s_loss_score_t_0_nt_128_p_11/eval_model_spec/{args.partition}_{args.eval_vlm}/results_dataframe.csv"
    elif args.filter_dataset == 'llava-mix':
        results_path = f'{root}/{args.filter_dataset}/ed_MODEL_SPEC_UPDATED_LLM_fd_llava-mix_ev_{args.eval_vlm}_wn_orig_neg_s_loss_score_t_0_nt_128_p_11/eval_model_spec/llava-mix_{args.eval_vlm}/results_dataframe.csv'
    else:
        print('filter dataset invalid.')

    eval_results = pd.read_csv(results_path)
    # eval_results.insert(len(eval_results.columns), "label", [''] * len(eval_results))

    for ind, item in eval_results.iterrows():

        # prepare prompt and set of ids to skip
        prompt, skip_ids = prepare_prompt(args.output_path, args.filter_dataset)
        if item['original_id'] in skip_ids:
            continue

        if args.num_samples and ind >= args.num_samples:
            break

        save_fn = f"{label_path}/{args.base_model.split('/')[-1]}_sample-{ind:08d}.csv"
        if os.path.exists(save_fn):
            continue
        else:
            # first create empty csv to use filename for minimizing duplicates when running runme_prompt
            pd.DataFrame([]).to_csv(save_fn)

        print('-' * 66)
        # print(f'Sample index {ind}\n')
        print(f'Original sample index {item["original_id"]}\n')

        pos = item['correct_option']
        # print('original positive: ', pos)

        neg = item['new_neg']
        # print('original negative: ', neg)

        input = prompt + f'\n\nQuestion: {item["prompt"]}\nS1:{pos}\nS2:{neg}\nResult:'

        generated = None
        while not generated:
            response = list(
                client.text.generation.create(
                    model_id=args.base_model,
                    inputs=[input],
                    parameters=params
                )
            )

            if response:
                # first generate new text
                response_text = response[0].results[0].generated_text.strip().split('\n')
                result = response_text[0].strip(' ')
                generated = result

                curr_df = pd.DataFrame([item])
                curr_df.insert(len(curr_df.columns), "label", [result])
                curr_df.to_csv(save_fn)
                # eval_results.at[ind, 'label'] = result
                # eval_results.to_csv(f'{args.output_path}/labels.csv')

                print(f'result: {result}')

    # eval_results.to_csv(f'{args.output_path}/labels.csv')
    return

if __name__ == '__main__':

    # if wanting to use in debugging mode
    # import pydevd_pycharm
    # debug_ip = os.environ.get('SSH_CONNECTION', None)
    # pydevd_pycharm.settrace(debug_ip, port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)

    # output directory
    job = time.strftime("%Y%m%d_%H%M")
    OUTPUT_DIR = f"/dccstor/leonidka1/irenespace/llm_results_updated/negs/{job}"
    label(output_path=OUTPUT_DIR)
    print('Script finished.')