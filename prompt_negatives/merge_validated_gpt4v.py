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

def validate(
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
    parser.add_argument('--debug',
                        action='store_true',
                        help='to debug')

    args = parser.parse_args()

    # if wanting to use in debugging mode
    if args.debug:
        import pydevd_pycharm
        debug_ip = os.environ.get('SSH_CONNECTION', None)
        pydevd_pycharm.settrace(debug_ip, port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)

    # get all the gpt4v validation results dataframes
    csv_name = 'gpt4v_all-samples.csv'
    partitions = {}
    if args.eval_dataset == 'SUGAR':
        partitions = {'replace_att_333', 'replace_obj_333', 'replace_rel_333'}
    elif args.eval_dataset == 'LN_SUBSET':
        partitions = {'ln_subset'}
    else:
        print(f'Implementation for dataset {args.eval_dataset} not yet implemented.')
    part_to_df = {p: pd.read_csv(f'{args.output_path}/step8/{p}/llava-v1.6-7b/{csv_name}') for p in partitions}

    for partition, df in part_to_df.items():
        df_valid = df[df['is_valid_sample'] == True] # restrict to validated samples

        # merge on the matching question, correct, and neg columns
        for model in {'llava-v1.6-7b', 'llava-v1.5-7b', 'instructblip_flan_t5', 'instructblip_vicuna_7b'}:
            df_model = pd.read_csv(f'{args.output_path}/step7/{partition}/{model}/filter_results_dataframe.csv')
            df_merged = pd.merge(
                df_valid, df_model, how='left',
                on=['question','correct','neg','q_ind','n_ind'],
                suffixes=["_gpt4v", f"_{model}"]
            )
            df_merged.to_csv(f'{args.output_path}/step8/{partition}/validated_{model}.csv')

    return

if __name__ == '__main__':
    validate()
    print('Script finished.')