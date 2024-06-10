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

# load_dotenv()
# API_KEY = os.getenv("OPENAI_KEY", None)
# HEADERS = {
#   "Content-Type": "application/json",
#   "Authorization": f"Bearer {API_KEY}"
# }

VLM_dict = {
    # 'llava-v1.6-7b': 'LLaVA 1.6-7b',
    # 'llava-v1.5-7b': 'LLaVA 1.5-7b',
    # 'instructblip_flan_t5': 'InstructBLIP Flan-T5',
    # 'instructblip_vicuna_7b': 'InstructBLIP Vicuna-7b',
    'internlm_xcomposer2_vl_7b': 'InternLM-XComposer2-VL-7b',
    'idefics2-8b': 'Idefics2-8b',
    'gpt4v': 'GPT-4V'
}

def filter(
):

    # model args
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path",
                        default='/dccstor/leonidka1/irenespace/gpt4v_results/desc/',
                        type=str,
                        help="Path to output directory.")
    parser.add_argument("--job_name",
                        default="1",
                        type=str,
                        help="Job name that is used in output directory path.")
    parser.add_argument("--select_by",
                        default="loss_score",
                        type=str,
                        help="Which type of inference mode was used during evaluation.")
    parser.add_argument("--step",
                        default="step7",
                        type=str,
                        help="Step of the pipeline, either step4 or step7.")
    parser.add_argument("--filter_count",
                        default=1,
                        type=int,
                        help="Number of models which answer incorrectly for each sample, to use as a filter criteria.")
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
    # if not os.path.exists(args.output_path):
    #     os.makedirs(args.output_path)
    #     print(f"output dir: {args.output_path}")

    # seed
    # SEED=42
    # print(f'Setting seed to: {SEED}')
    # np.random.seed(SEED)
    # random.seed(SEED)

    # get all the perplexity results dataframes for this iteration of questions
    vlm_dfs = {vlm: pd.read_csv(f'{args.output_path}/{args.step}/{args.job_name}/{vlm}/{args.select_by}/results_dataframe.csv').sort_values(by=['orig_id','q_ind','n_ind']) for vlm in VLM_dict}
      # this length should be equal for all dfs

    for vlm, df in vlm_dfs.items():
        df_llava = pd.read_csv(f'{args.output_path}/{args.step}/{args.job_name}/llava-v1.6-7b/{args.select_by}/filter_results_dataframe_count-{args.filter_count}.csv').sort_values(by=['orig_id','q_ind','n_ind'])
        total_samples = len(df_llava)

        df_merged = pd.merge(df_llava, df, how='left', on=['orig_id', 'q_ind', 'n_ind'], suffixes=['_llava-v1.6-7b', f'_{vlm}'])
        assert len(df_merged) == total_samples
        df_merged.to_csv(f'{args.output_path}/{args.step}/{args.job_name}/{vlm}/{args.select_by}/filter_results_dataframe_count-{args.filter_count}.csv')

    return

if __name__ == '__main__':
    # output directory
    # job = time.strftime("%Y%m%d_%H%M")
    # OUTPUT_DIR = f"/dccstor/leonidka1/irenespace/llm_results_updated/negs/{job}"
    filter()
    print('Script finished.')