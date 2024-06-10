import time
import json
import os
import sys
import random
import re
import string
import argparse
from functools import partial
import ast

# import fire
import tqdm
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

from dotenv import load_dotenv
from genai.credentials import Credentials
from genai.model import Model
from genai.schemas import GenerateParams
from genai.schemas.generate_params import LengthPenalty

from dec_vl_eval.src.datasets.benchmark_datasets import get_benchmark_dataset

# load_dotenv()
# API_KEY = os.getenv("GENAI_KEY", None)
# API_URL = os.getenv("GENAI_API", None)
#
# def classify_contradictions(
#         contradictions,
#         check_generator
# ):
#     '''
#     helper function to return any differences between two given texts
#
#     :param inp1: original data sample (i.e. original positive or negative)
#     :param inp2: negative option for data sample (i.e. corresponding new negative or original negative)
#     :param check_generator: model to use for generating the response
#     :return: string of llm-classified partition labels
#     '''
#     input = "Input: {'contradictions': [{'S1': 'black shirt', 'S2': 'white shirt'}]}\nOutput: attribute\n\n" \
#             "Input: {'contradictions': [{'S1': 'ball on the table', 'S2': 'ball under the table'}]}\nOutput: relation\n\n" \
#             "Input: {'contradictions': [{'S1': 'black bag', 'S2': 'black car'}]}\nOutput: object\n\n" \
#             "Input: {'contradictions': [{'S1': 'meat sandwich', 'S2': 'vegetarian sandwich'}]}\nOutput: attribute\n\n" \
#             f"Input: {contradictions}\nOutput:"
#     response = check_generator.generate([input])
#
#     if not response:
#         raise ValueError("Error: generated difference results in None.")
#
#     return response[0].generated_text.split('\n')[0].strip()
#
# def check_contradict(
#         inp1,
#         inp2,
#         check_generator
# ):
#     '''
#     helper function to return any contradictions between two given texts
#
#     :param inp1: original data sample (i.e. original positive or negative)
#     :param inp2: generated new data sample (i.e. corresponding new positive or negative)
#     :param check_generator: model to use for generating the response
#     :return: string of any detected contradictions
#     '''
#     contradict_input = 'You are given two image descriptions to compare. Your task is to identify all contradictions, or differences in meaning, between these two descriptions. ' \
#                        'If two phrases use different words but one is just a generalized version of the other or if the two phrases could logically mean the same thing, then it does not suffice as a contradiction. ' \
#                        'Return the list of contradictions in json format.\n\nHere are some examples:\n\n' \
#                        'S1: "Pen is to the left of the case"\nS2: "To the left of the case is a pen"\nResult:\n{"contradictions": []}\n\n' \
#                        'S1: "In the sunlight, a flock of bats flies over the ocean towards the island"\nS2: "A flock of seagulls flies above the ocean and away from the island in the sunlight"\n' \
#                        'Result:\n{"contradictions": [{"S1": "bats", "S2": "seagulls"}, {"S1": "towards the island", "S2": "away from the island"}]}\n\n' \
#                        'S1: "A family of birds perched on a branch"\nS2: "A group of birds perched beside a branch"\nResult:\n{"contradictions": []}\n\n' \
#                        'S1: "A tall man is jumping on a red sofa"\nS2: "A short man sits on a blue sofa"\nResult:\n' \
#                        '{"contradictions": [{"S1": "jumping on a sofa", "S2": "sits on a sofa"}, {"S1": "sofa is red", "S1": "sofa is blue"}, {"S1": "man is tall", "S2": "man is short"}]}' \
#                        f'\n\nS1: "{inp1}"\nS2: "{inp2}"\nResult:\n'
#     contradict_response = check_generator.generate([contradict_input])
#
#     if not contradict_response:
#         raise ValueError("Error: generated contradiction results in None.")
#
#     return contradict_response[0].generated_text.split('\n')[0].strip()


def prompt(
    args,
    seed=42,
):

    # get requested dataset by using helper function from benchmarks
    data_dict, options = get_benchmark_dataset(args=args)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # creds = Credentials(args.genai_key, api_endpoint=API_URL)
    # check_params = GenerateParams(
    #     decoding_method="sample",
    #     max_new_tokens=128,
    #     temperature=0.5,
    #     top_p=1.0,
    #     random_seed=seed,
    #     repetition_penalty=1.0,
    # )
    # check_generator = Model(args.base_model, params=check_params, credentials=creds)
    #
    # classify_params = GenerateParams(
    #     decoding_method="greedy",
    #     max_new_tokens=5
    # )
    # classify_generator = Model("meta-llama/llama-2-70b", params=classify_params, credentials=creds)

    for key, samples in data_dict.items():

        if args.dataset_partition and key != args.dataset_partition:
            continue

        output_dir = f"/dccstor/leonidka1/irenespace/llm_results/sim_rouge" + (
            f"/{args.dataset_partition}" if args.dataset_partition else "") + f"/{args.job_name}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"output dir: {output_dir}")

        for ind, sample in enumerate(samples):
            if args.num_samples and ind == args.num_samples:
                break

            save_fn = f"{output_dir}/rouge_{key}_sample-{ind:04d}.csv"
            if os.path.exists(save_fn):
                continue
            else:
                # first create empty csv to use filename for minimizing duplicates when running runme_prompt
                pd.DataFrame([]).to_csv(save_fn)

            print('-' * 66)
            print(f'Sample index {ind}\n')

            orig_ind = int(sample[4])
            print(f'Original sample index {orig_ind}\n')

            pos = sample[2]
            print('original positive: ', pos)

            neg = sample[6]
            print('original negative: ', neg)

            new_neg = sample[7]
            print('new negative: ', new_neg)

            # if args.choice == 'orig':
            #     to_classify = neg
            #
            #     contradictions = None
            #     while not contradictions:
            #         contradict_text = check_contradict(pos, neg, check_generator)
            #         print('contradict response: ', contradict_text)
            #
            #         contradict_dict = json.loads(contradict_text)
            #
            #         if "contradictions" not in contradict_dict:
            #             continue
            #
            #         contradictions = contradict_dict
            #
            # elif args.choice == 'new':
            #     to_classify = new_neg
            #
            #     contradictions = ast.literal_eval(sample[9])
            # else:
            #     raise ValueError('Please specify a valid classification choice.')
            #
            # print('Contradictions for input: ', contradictions)
            #
            # gen_responses = []
            # for item_ind, item in enumerate(contradictions["contradictions"]):
            #     curr_contradict = "{'contradictions': [" + str(item) + "]}"
            #     curr_response = classify_contradictions(curr_contradict, classify_generator)
            #     if curr_response:
            #         gen_responses.append(curr_response)
            #     else:
            #         raise ValueError("Error: generated result in None.")
            #     print(f'response for contradiction {item_ind}: ', curr_response)

            # save file
            orig_rouge = scorer.score(pos, neg)
            new_rouge = scorer.score(pos, new_neg)
            curr_dict = {
                'orig_ind': int(orig_ind),  # original index from the loaded dataset
                'image_path': sample[0],  # original image path
                'orig_pos': pos,  # original positive
                'orig_neg': neg,  # original negative
                'new_neg': new_neg,
                'new_change': sample[8],
                'contradiction_check_llm': sample[9],
                'orig_rouge1': orig_rouge['rouge1'],
                'orig_rouge2': orig_rouge['rouge2'],
                'orig_rougeL': orig_rouge['rougeL'],
                'new_rouge1': new_rouge['rouge1'],
                'new_rouge2': new_rouge['rouge2'],
                'new_rougeL': new_rouge['rougeL'],
            }

            pd.DataFrame([curr_dict]).to_csv(save_fn)
            # print(f'finished sample {ind}: saved filename for original sample {orig_ind} at {save_fn}')

    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # model args
    # parser.add_argument("--base_model",
    #                     default='tiiuae/falcon-180b',
    #                     type=str,
    #                     help="Which language model to use.")

    # dataset args
    parser.add_argument("--eval_dataset",
                        default='COMBINED_LLM',
                        type=str,
                        help="Which dataset to use for evaluation.")
    parser.add_argument("--dataset_partition",
                        default=None,
                        type=str,
                        help="Which partition of the dataset, if any.")
    parser.add_argument("--num_samples",
                        default=None,
                        type=int,
                        help="Optional number of samples to limit from the dataset.")
    # parser.add_argument("--choice",
    #                     default='orig', # orig or new
    #                     type=str,
    #                     help="Which negative option to use, either the original or the new generated one.")
    parser.add_argument("--select_by",
                        default='loss_score',
                        type=str,
                        help="Which evaluation method to use.")

    parser.add_argument('--debug',
                        action='store_true',
                        help='to debug')

    parser.add_argument("--job_name",
                        default=time.strftime("%Y%m%d_%H%M"),
                        type=str,
                        help="job name")

    # generation args
    # parser.add_argument("--genai_key",
    #                     default=API_KEY,
    #                     type=str,
    #                     help="Which api key to use for prompting.")
    # parser.add_argument("--num_responses",
    #                     default=1,
    #                     type=int,
    #                     help="Number of responses to generate per sample.")

    args = parser.parse_args()

    if args.debug:
        import pydevd_pycharm

        debug_ip = os.environ.get('SSH_CONNECTION', None)
        pydevd_pycharm.settrace(debug_ip, port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)

    prompt(args)
