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
# from openai import OpenAI
# from dotenv import load_dotenv

# import fire
import tqdm
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

from dec_vl_eval.src.datasets.benchmark_datasets import get_benchmark_dataset

def randomly_sample(
):

    # model args
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path",
                        default='/dccstor/leonidka1/irenespace/gpt4v_results/desc/',
                        type=str,
                        help="Path to output directory.")
    parser.add_argument("--eval_dataset",
                        default="SUGAR",
                        type=str,
                        help="Which dataset was used.")
    parser.add_argument("--select_by",
                        default="loss_score",
                        type=str,
                        help="Which type of inference mode was used during evaluation.")
    parser.add_argument("--step",
                        default="step7",
                        type=str,
                        help="Step of the pipeline, either step4 or step7.")
    parser.add_argument("--num_samples",
                        default=100,
                        type=int,
                        help="# of samples to randomly select")
    parser.add_argument("--results_type",
                        default=None,
                        type=str,
                        help="either None or 'filter', to denote which type of results dataframe to use")
    parser.add_argument("--filter_count",
                       default=1,
                       type=int,
                       help="Number of models to use for filter criteria")
    parser.add_argument("--is_incorrect",
                        action='store_true',
                        help="to filter for incorrectly answered samples first")
    parser.add_argument("--model",
                        default="llava-v1.6-7b",
                        type=str,
                        help="Which model to process errors for.")
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
    csv_name = ('filter_' if (args.results_type and args.results_type == 'filter') else '') + 'results_dataframe' + (f'_count-{args.filter_count}' if (args.filter_count > 1) else '') + '.csv'
    if args.eval_dataset == 'SUGAR':
        part_to_df = {p: pd.read_csv(f'{args.output_path}/{args.step}/{p}_333/{args.model}/{args.select_by}/{csv_name}') for p in
                   {'replace_att', 'replace_obj', 'replace_rel'}}
    elif args.eval_dataset == 'LN_SUBSET':
        part_to_df = {m: pd.read_csv(f'{args.output_path}/{args.step}/ln_subset/{m}/{args.select_by}/{csv_name}') for m in
                   {'llava-v1.6-7b', 'llava-v1.5-7b', 'instructblip_flan_t5', 'instructblip_vicuna_7b'}}
    elif args.eval_dataset == 'LN_LESS_NOUNS':
        part_to_df = {m: pd.read_csv(f'{args.output_path}/{args.step}/ln_less_nouns/{m}/{args.select_by}/{csv_name}') for m in
                   {'llava-v1.6-7b', 'llava-v1.5-7b', 'instructblip_flan_t5', 'instructblip_vicuna_7b'}}
    else:
        print(f'Implementation for dataset {args.eval_dataset} not yet implemented.')

    save_csv_name = ('incorrect_only_' if args.is_incorrect else '') + csv_name
    for key, df in part_to_df.items():
        curr_df = df.copy()
        if args.is_incorrect:
            curr_df = curr_df[curr_df['accuracy' + (f'_{args.model}' if args.filter_count > 1 else '')] == 0]
        new_df = curr_df.sample(n=args.num_samples, random_state=SEED)

        if args.eval_dataset == 'SUGAR':
            new_df.to_csv(f'{args.output_path}/{args.step}/{key}_333/{args.model}/{args.select_by}/error_rate_{save_csv_name}')
        elif args.eval_dataset == 'LN_SUBSET':
            new_df.to_csv(f'{args.output_path}/{args.step}/ln_subset/{key}/{args.select_by}/error_rate_{save_csv_name}')
        elif args.eval_dataset == 'LN_LESS_NOUNS':
            new_df.to_csv(f'{args.output_path}/{args.step}/ln_less_nouns/{key}/{args.select_by}/error_rate_{save_csv_name}')
    return

if __name__ == '__main__':
    randomly_sample()
    print('Script finished.')