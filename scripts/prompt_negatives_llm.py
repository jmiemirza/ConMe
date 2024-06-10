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

def prompt_negative(inp, partition):
    '''
    helper function for constructing the prompt to generate a hard negative

    :param inp: string for the original positive
    :return: string of the formatted prompt
    '''

    partition2prompt = {
        'replace_obj': 'You are given an image annotation. Your task is to modify the meaning by replacing one of the existing objects. ' \
                       'List both the change of object and new text.\n\nAdditionally, the new text must satisfy all of the following three requirements:\n' \
                       '1. The new text must now be inaccurate of the image.\n' \
                       '2. Compared to the original text, the new text must differ in only one object. All other details must be kept the same.\n' \
                       '3. The new text must be grammatically correct, fluent, and sensible.\n\nHere are some examples:\n\n' \
                       'Original text:  "To the left of the table, a man lays down to read a book"\nChange of object: "book" --> "magazine"\n' \
                       'New text: "To the left of the table, a man lays down to read a magazine"\n\n' \
                       'Original text: "An apple rolls on top of the kitchen table"\nChange of object: "apple" --> "orange"\n' \
                       'New text: "An orange rolls on top of the kitchen table"\n\n' \
                       'Original text: "A couple students stand by the edge of the Charles River, which has frozen over"\nChange of object: "Charles River" --> "pond"\n' \
                       'New text: "A couple students stand by the edge of the pond, which has frozen over"\n\n' \
                       f'Original text: "{inp}"',
        'replace_att': 'You are given an image annotation. Your task is to modify the meaning by replacing one of the existing attributes. ' \
                       'List both the change of attribute and new text.\n\nAdditionally, the new text must satisfy all of the following three requirements:\n' \
                       '1. The new text must now be inaccurate of the image.\n' \
                       '2. Compared to the original text, the new text must differ in only one attribute. All other details must be kept the same.\n' \
                       '3. The new text must be grammatically correct, fluent, and sensible.\n\nHere are some examples:\n\n' \
                       'Original text:  "To the left of the red sofa, a man lays down to read a book"\nChange of attribute: "red" --> "orange"\n' \
                       'New text: "To the left of the orange sofa, a man lays down to read a book"\n\n' \
                       'Original text: "An apple rolls on top of the kitchen table"\nChange of attribute: "an" --> "multiple"\n' \
                       'New text: "Multiple apples roll on top of the kitchen table"\n\n' \
                       'Original text: "A couple students stand by the edge of the river, which has frozen over"\nChange of attribute: "has frozen over" --> "flows in rushing waves"\n' \
                       'New text: "A couple students stand by the edge of the river, which flows in rushing waves"\n\n' \
                       f'Original text: "{inp}"',
        'replace_rel': 'You are given an image annotation. Your task is to modify the meaning by replacing one of the existing spatial relations. ' \
                       'List both the change of spatial relation and new text.\n\nAdditionally, the new text must satisfy all of the following three requirements:\n' \
                       '1. The new text must now be inaccurate of the image.\n' \
                       '2. Compared to the original text, the new text must differ in only one spatial relation. All other details must be kept the same.\n' \
                       '3. The new text must be grammatically correct, fluent, and sensible.\n\nHere are some examples:\n\n' \
                       'Original text:  "The dining table near the kitchen has a bowl of fruit on it"\nChange of spatial relation: "near" --> "inside"\n' \
                       'New text: "The dining table inside the kitchen has a bowl of fruit on it”"\n\n' \
                       'Original text: "An apple rolls on top of the kitchen table"\nChange of spatial relation: "on top of" --> "off of"\n' \
                       'New text: "A red apple rolls off of the kitchen table"\n\n' \
                       'Original text: "To the left of the red sofa, a man lays down to read a book"\nChange of spatial relation: "to the left of" --> "in front of"\n' \
                       'New text: "In front of the red sofa, a man sits reading a book"\n\n' \
                       f'Original text: "{inp}"',

    }

    return partition2prompt[partition]


def check_contradict(
        inp1,
        inp2,
        check_params,
        llm,
        client
):
    '''
    helper function to return any contradictions between two given texts

    :param inp1: original data sample (i.e. original positive or negative)
    :param inp2: generated new data sample (i.e. corresponding new positive or negative)
    :param check_params: parameters to use for generating the response
    :return: string of any detected contradictions
    '''
    contradict_input = 'You are given two image annotations, S1 and S2, to compare. Your task is to identify all contradictions, or differences in meaning, between these two descriptions. ' \
                       'If two phrases use different words which have similar meaning or if one is just a generalized version of the other, then it does not suffice as a contradiction. ' \
                       'Return the list of contradictions in json format.\n\nHere are some examples:\n\n' \
                       'S1: "Pen is to the left of the case"\nS2: "To the left of the case is the pen"\nResult:\n{"contradictions": []}\n\n' \
                       'S1: "In the sunlight, a flock of bats flies over the ocean towards the island"\nS2: "A flock of seagulls flies above the ocean and away from the island in the sunlight"\n' \
                       'Result:\n{"contradictions": [{"S1": "bats", "S2": "seagulls"}, {"S1": "towards the island", "S2": "away from the island"}]}\n\n' \
                       'S1: "A family of birds perched on a branch"\nS2: "A group of birds perched beside a branch"\nResult:\n{"contradictions": []}\n\n' \
                       'S1: "A tall man is jumping on a red sofa"\nS2: "A short man sits on a blue sofa"\nResult:\n' \
                       '{"contradictions": [{"S1": "jumping on a sofa", "S2": "sits on a sofa"}, {"S1": "sofa is red", "S1": "sofa is blue"}, {"S1": "man is tall", "S2": "man is short"}]}' \
                       f'\n\nS1: "{inp1}"\nS2: "{inp2}"\nResult:'
    # contradict_response = check_generator.generate([contradict_input])

    contradict_response = list(
        client.text.generation.create(
            model_id=llm,
            inputs=[contradict_input],
            parameters=check_params
        )
    )

    if not contradict_response:
        raise ValueError("Error: generated contradiction results in None.")

    return contradict_response[0].results[0].generated_text.strip().split('\n')[0].strip()


def prompt(
    # base_model: str,
    # dataset: str,
    output_path: str,
    # partition=None,
    temperature=1.0,
    top_k=100,
    top_p=1.0,
    seed=42,
    # max_seq_len: int = 512,
    generate_max_len: int = 200,
):

    # model args
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",
                        default='ibm-mistralai/mixtral-8x7b-instruct-v01-q',
                        type=str,
                        help="Which language model to use.")

    # dataset args
    parser.add_argument("--eval_dataset",
                        default='SUGAR',
                        type=str,
                        help="Which dataset to use for evaluation.")
    parser.add_argument("--partition",
                        default=None,
                        type=str,
                        help="Which partition of the dataset, if any.")
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
    parser.add_argument('--debug',
                        action='store_true',
                        help='to debug')

    # generation args
    parser.add_argument("--genai_key",
                        default=API_KEY,
                        type=str,
                        help="Which BAM api key to use.")
    parser.add_argument("--num_responses",
                        default=5,
                        type=int,
                        help="Number of responses to generate per sample.")

    args = parser.parse_args()

    # if wanting to use in debugging mode
    if args.debug:
        import pydevd_pycharm
        debug_ip = os.environ.get('SSH_CONNECTION', None)
        pydevd_pycharm.settrace(debug_ip, port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)

    # check that output path dir exists
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        print(f"output dir: {args.output_path}")

    # get requested dataset by using helper function from benchmarks
    data_dict, options = get_benchmark_dataset(args=args)

    creds = Credentials(args.genai_key, api_endpoint=API_URL)
    params = TextGenerationParameters(
        decoding_method=DecodingMethod.SAMPLE,
        max_new_tokens=generate_max_len,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        # random_seed=seed,
        repetition_penalty=1.0,
        # length_penalty=LengthPenalty(decay_factor=1.3)
        # truncate_input_tokens=max_seq_len
    )

    client = Client(credentials=creds)
    # generator = Model(args.base_model, params=params, credentials=creds)

    # check_generator model for double checking contradictions and differences between
    # original and generated samples; parameters use temp=0.5 for a more restricted output
    check_params = TextGenerationParameters(
        decoding_method=DecodingMethod.SAMPLE,
        max_new_tokens=128,
        temperature=0.5,
        top_p=1.0,
        random_seed=seed,
        repetition_penalty=1.0,
        # length_penalty=LengthPenalty(decay_factor=1.3)
        # truncate_input_tokens=max_seq_len
    )
    # check_generator = Model(args.base_model, params=check_params, credentials=creds)

    # seed
    max_int_32bit = 2 ** 32 - 1
    SEED = int(round(time.time() * 1000)) % max_int_32bit
    print(f'Setting seed to: {SEED}')
    np.random.seed(SEED)
    random.seed(SEED)

    for key, samples in data_dict.items():

        all_samples = random.sample(samples, k=len(samples))

        if args.eval_dataset == 'SUGAR' and args.partition and key != args.partition:
            continue

        for ind, sample in enumerate(all_samples):
            if args.num_samples and ind == args.num_samples:
                break

            save_fn = f"{args.output_path}/{args.base_model.split('/')[-1]}_{key}_sample-{ind:04d}.csv"
            if os.path.exists(save_fn):
                continue
            else:
                # first create empty csv to use filename for minimizing duplicates when running runme_prompt
                pd.DataFrame([]).to_csv(save_fn)

            print('-' * 66)
            print(f'Seeded sample index {ind}\n')
            print(f'Original sample index {sample[-1]}\n')

            pos = sample[2]
            print('original positive: ', pos)

            neg = [text for text in sample[-2] if text != pos][0]
            # print('original negative: ', neg)

            include = True
            gen_prompts = []
            gen_changes = []
            gen_contradicts = []

            for response_ind in range(args.num_responses):

                # keep track of how many generation attempts made; skip this data sample
                # once this reaches 3 futile attempts
                attempts = 0
                while len(gen_prompts) < response_ind + 1:

                    # response = generator.generate([prompt_negative(pos, args.partition)])
                    response = list(
                        client.text.generation.create(
                            model_id=args.base_model,
                            inputs=[prompt_negative(pos, args.partition)],
                            parameters=params
                        )
                    )

                    if response:
                        # first generate new text
                        response_text = response[0].results[0].generated_text.strip().split('\n')

                        try:
                            text_change = response_text[0].strip(' ')
                            # print(f'changed text {response_ind}: ', text_change)

                            new_neg = response_text[1].strip(' ').split('"')[1]
                            print(f'new negative {response_ind}: ', new_neg)

                            if new_neg in gen_prompts: # avoid duplicate negatives
                                # print('This neg was already generated.')
                                continue

                            # check for contradictions, which should not be empty, since we are generating hard negatives
                            contradict_text = check_contradict(pos, new_neg, check_params, args.base_model, client)
                            # print('contradict response: ', contradict_text)

                            contradict_dict = json.loads(contradict_text)

                            if not contradict_dict['contradictions']:
                                attempts += 1
                                print('number of futile attempts: ', attempts)
                                if attempts == 3:
                                    include = False
                                    break
                                continue

                        except json.decoder.JSONDecodeError:
                            # print('Encountered json error in parsing model responses --> regenerating response')
                            continue

                        except IndexError:
                            # print('Model generated negative not in expected format --> regenerating response')
                            continue

                        except KeyError:
                            # print('Model identified contradictions not in expected format --> regenerating response')
                            continue

                        gen_prompts.append(new_neg)
                        gen_changes.append(text_change)
                        gen_contradicts.append(contradict_dict)

                    else:
                        raise ValueError("Error: generated result in None.")

                if not include:
                    break

            if include:
                # save file
                curr_dict = {
                    'orig_ind': sample[-1], # original index from the loaded dataset
                    'image_path': sample[0], # original image path
                    'orig_pos': pos, # original positive
                    'orig_neg': neg, # original negative
                }
                for pos_ind, new_pos in enumerate(gen_prompts):
                    curr_dict[f'new_neg_{pos_ind}'] = new_pos
                    curr_dict[f'new_change_{pos_ind}'] = gen_changes[pos_ind]
                    curr_dict[f'new_contradict_{pos_ind}'] = gen_contradicts[pos_ind]

                pd.DataFrame([curr_dict]).to_csv(save_fn)
                print(f'finished seeded sample {ind}: saved filename for original sample {sample[-1]} at {save_fn}')

    return

if __name__ == '__main__':

    # if wanting to use in debugging mode
    # import pydevd_pycharm
    # debug_ip = os.environ.get('SSH_CONNECTION', None)
    # pydevd_pycharm.settrace(debug_ip, port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)

    # output directory
    job = time.strftime("%Y%m%d_%H%M")
    OUTPUT_DIR = f"/dccstor/leonidka1/irenespace/llm_results_updated/negs/{job}"
    prompt(output_path=OUTPUT_DIR)
    print('Script finished.')