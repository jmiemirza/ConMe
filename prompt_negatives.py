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
import torch
import tqdm
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
#
# from dotenv import load_dotenv
# from genai import Credentials, Client
# # from genai.model import Model
# from genai.schema import TextGenerationParameters, TextGenerationReturnOptions, DecodingMethod
# # from genai.schemas.generate_params import LengthPenalty

from dec_vl_eval.src.datasets.benchmark_datasets import get_benchmark_dataset
from dec_vl_eval.src.utils.directory import option_prompt_dict
from dec_vl_eval.src.models import get_model, get_caption_model, model_generate, model_add_captions, score_options

# llava dependencies
from dec_vl_eval.src.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from dec_vl_eval.src.llava.conversation import conv_templates, SeparatorStyle
from dec_vl_eval.src.llava.model.builder import load_pretrained_model
from dec_vl_eval.src.llava.utils import disable_torch_init
from dec_vl_eval.src.llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from dec_vl_eval.src.utils.utils_llava import load_image, str_to_remove, compute_image_tensor_llava

# t2i_metric dependencies
# from dec_vl_eval.src.t2i_metrics import get_score_model

# load_dotenv()
# API_KEY = os.getenv("GENAI_KEY", None)
# API_URL = os.getenv("GENAI_API", None)

def get_full_prompt(conv_mode, qs, mm_use_im_start_end):
    """
    add system prompt,  and  <image>,  USER:,  ASSISTANT:
    e.g.

    A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>
    What is the color of the chair seen on the right side of the image? ASSISTANT:


    """
    if mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt, conv


def prompt_negative(inp, partition):
    '''
    helper function for constructing the prompt to generate a hard negative

    :param inp: string for the original positive
    :return: string of the formatted prompt
    '''

    partition2prompt = {
        'replace_obj': 'You are given an image and a corresponding image annotation. Your task is to modify the meaning by replacing one of the existing objects. ' \
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
        'replace_att': 'You are given an image and a corresponding image annotation. Your task is to modify the meaning by replacing one of the existing attributes. ' \
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
        'replace_rel': 'You are given an image and a corresponding image annotation. Your task is to modify the meaning by replacing one of the existing spatial relations. ' \
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
        # check_params,
        # llm,
        # client
):
    '''
    helper function to return any contradictions between two given texts

    :param inp1: original data sample (i.e. original positive or negative)
    :param inp2: generated new data sample (i.e. corresponding new positive or negative)
    :param check_params: parameters to use for generating the response
    :return: string of any detected contradictions
    '''
    # contradict_input = 'You are given an image and two image annotations, S1 and S2, to compare. Your task is to identify all contradictions, or differences in meaning, between these two annotations. ' \
    #                    'If two phrases use different words which have similar meaning or if one is just a generalized version of the other, then it does not suffice as a contradiction. ' \
    #                    'Return the list of contradictions in json format.\n\nHere are some examples:\n\n' \
    #                    'S1: "Pen is to the left of the case"\nS2: "To the left of the case is the pen"\nResult:\n{"contradictions": []}\n\n' \
    #                    'S1: "In the sunlight, a flock of bats flies over the ocean towards the island"\nS2: "A flock of seagulls flies above the ocean and away from the island in the sunlight"\n' \
    #                    'Result:\n{"contradictions": [{"S1": "bats", "S2": "seagulls"}, {"S1": "towards the island", "S2": "away from the island"}]}\n\n' \
    #                    'S1: "A family of birds perched on a branch"\nS2: "A group of birds perched beside a branch"\nResult:\n{"contradictions": []}\n\n' \
    #                    'S1: "A tall man is jumping on a red sofa"\nS2: "A short man sits on a blue sofa"\nResult:\n' \
    #                    '{"contradictions": [{"S1": "jumping on a sofa", "S2": "sits on a sofa"}, {"S1": "sofa is red", "S1": "sofa is blue"}, {"S1": "man is tall", "S2": "man is short"}]}' \
    #                    f'\n\nS1: "{inp1}"\nS2: "{inp2}"\nResult:'
    contradict_input = 'You are given two image annotations, S1 and S2, to compare. Your task is to identify and list contradictions, or differences in meaning, if any exist, between these two annotations. ' \
                       'Include only the identified contradictions, if any exist, in your answer, and answer with a json-formatted list.\n\nHere are some examples:\n\n' \
                       'S1: "Pen is to the left of the case"\nS2: "To the left of the case is the pen"\nResult:\n{"contradictions": []}\n\n' \
                       'S1: "In the sunlight, a flock of bats flies over the ocean towards the island"\nS2: "A flock of seagulls flies above the ocean and away from the island in the sunlight"\n' \
                       'Result:\n{"contradictions": [{"S1": "bats", "S2": "seagulls"}, {"S1": "towards the island", "S2": "away from the island"}]}\n\n' \
                       'S1: "Pen is to the left of the case"\nS2: "To the right of the case is the pen"\nResult:\n{"contradictions": [{"S1": "to the left", "S2": "to the right"}]}\n\n' \
                       'S1: "A family of birds perched on a branch"\nS2: "A group of birds perched beside a branch"\nResult:\n{"contradictions": []}\n\n' \
                       'S1: "A tall man is jumping on a red sofa"\nS2: "A short man sits on a blue sofa"\nResult:\n' \
                       '{"contradictions": [{"S1": "jumping on a sofa", "S2": "sits on a sofa"}, {"S1": "sofa is red", "S1": "sofa is blue"}, {"S1": "man is tall", "S2": "man is short"}]}' \
                       f'\n\nS1: "{inp1}"\nS2: "{inp2}"\nResult:'
    return contradict_input

def llava_generate(
        full_instruction_prompt,
        conv,
        image_tensor,
        tokenizer,
        args,
        model
):
    input_ids = tokenizer_image_token(full_instruction_prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                      return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            # do_sample=True,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            # streamer=streamer,
            use_cache=True,
            stopping_criteria=[stopping_criteria]
        )
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().strip('\n')
    conv.messages[-1][-1] = outputs
    for str_ in str_to_remove:
        if str_ != '\n':
            outputs = outputs.replace(str_, '')
    return outputs

class DictObj:
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
               setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
               setattr(self, key, DictObj(val) if isinstance(val, dict) else val)

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
                        default='llava-v1.6-7b',
                        type=str,
                        help="Which VLM to use.")
    # parser.add_argument("--llm",
    #                     default='ibm-mistralai/mixtral-8x7b-instruct-v01-q',
    #                     type=str,
    #                     help="Which llm to use.")

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
    # parser.add_argument("--genai_key",
    #                     default=API_KEY,
    #                     type=str,
    #                     help="Which BAM api key to use.")
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get the model
    if args.base_model == 'llava-v1.5-7b':
        llava_args = {
            'model': 'llava-v1.5-7b',
            'model_path': "liuhaotian/llava-v1.5-7b",
            # 'model_path': "/dccstor/leonidka1/irenespace/data/llava_weights/llava-v1.5-7b",
            'model_base': None,
            'load_8bit': False,
            'load_4bit': False,
            'prompt': 11,
            'temperature': 1.0,
            'max_new_tokens': 128,
            'conv_mode': "llava_v1",
            'select_by': None,
            'tokenizer': None,
            'image_processor': None,
            'device': device,
            # 't2i_model': get_score_model('llava-v1.5-7b')
        }
        llava_args = DictObj(llava_args)
        tokenizer, model, image_processor = get_model(llava_args, device)  # llava-1.5

    elif args.base_model == 'llava-v1.6-7b':
        llava_args = {
            'model': 'llava-v1.6-7b',
            'model_path': "liuhaotian/llava-v1.6-vicuna-7b",
            'model_base': None,
            'load_8bit': False,
            'load_4bit': False,
            'prompt': 11,
            'temperature': 1.0,
            'max_new_tokens': 128,
            'conv_mode': "llava_v1",
            'select_by': None,
            'tokenizer': None,
            'image_processor': None,
            'device': device,
            # 't2i_model': get_score_model('llava-v1.6-7b')
        }
        # llava_args = DictObj(llava_args)
        tokenizer, model, image_processor = get_model(DictObj(llava_args), device)  # llava-1.6

    elif 'instructblip' in args.eval_vlm:
        vlm_args = {
            'backend': 'hf',
            'model': args.eval_vlm,
        }
        # if args.eval_vlm == 'instructblip_flant5':
        #     vlm_args['t2i_model'] = get_score_model('instructblip-flant5-xl')

        model, instructblip_combined_processors = get_model(DictObj(vlm_args), device)

    else:
        raise ValueError(f'No implementation for {args.eval_vlm} found.')

    # creds = Credentials(args.genai_key, api_endpoint=API_URL)
    # params = TextGenerationParameters(
    #     decoding_method=DecodingMethod.SAMPLE,
    #     max_new_tokens=generate_max_len,
    #     temperature=temperature,
    #     top_p=top_p,
    #     top_k=top_k,
    #     # random_seed=seed,
    #     repetition_penalty=1.0,
    #     # length_penalty=LengthPenalty(decay_factor=1.3)
    #     # truncate_input_tokens=max_seq_len
    # )
    #
    # client = Client(credentials=creds)

    # check_generator model for double checking contradictions and differences between
    # original and generated samples; parameters use temp=0.5 for a more restricted output
    # check_params = TextGenerationParameters(
    #     decoding_method=DecodingMethod.SAMPLE,
    #     max_new_tokens=128,
    #     temperature=0.5,
    #     top_p=1.0,
    #     random_seed=seed,
    #     repetition_penalty=1.0,
    #     # length_penalty=LengthPenalty(decay_factor=1.3)
    #     # truncate_input_tokens=max_seq_len
    # )

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
            print('original negative: ', neg)

            file_name = sample[0]
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
                    # response = list(
                    #     client.text.generation.create(
                    #         model_id=args.base_model,
                    #         inputs=[prompt_negative(pos, args.partition)],
                    #         parameters=params
                    #     )
                    # )

                    # generate negs
                    # letters_2options = ['A. ', 'B. ']

                    if 'llava-v1.5' in args.base_model or 'llava-v1.6' in args.base_model:
                        gen_args = DictObj(llava_args)
                        image, image_tensor = compute_image_tensor_llava(file_name, gen_args, model, image_processor)

                        extended_instruction = prompt_negative(pos, args.partition)
                        full_instruction_prompt, conv = get_full_prompt(conv_mode=gen_args.conv_mode,
                                                                        qs=extended_instruction,
                                                                        mm_use_im_start_end=model.config.mm_use_im_start_end)
                        response = llava_generate(full_instruction_prompt, conv, image_tensor, tokenizer, gen_args, model)

                    elif 'minigpt4v2' in args.base_model:
                        raise NotImplementedError("Didn't implement the generation inference for Minigpt4.")

                    else:
                        raise NotImplementedError("Didn't implement the generation inference for Instructblips.")

                    if response:
                        # first generate new text
                        response_text = response.strip().split('\n')

                        try:
                            text_change = response_text[0].strip(' ')
                            print(f'changed text {response_ind}: ', text_change)

                            new_neg = response_text[1].strip(' ').split('"')[1]
                            print(f'new negative {response_ind}: ', new_neg)

                            if new_neg.lower() in gen_prompts: # avoid duplicate negatives
                                # print('This neg was already generated.')
                                attempts += 1
                                print('number of futile attempts: ', attempts)
                                if attempts == 5:
                                    include = False
                                    break
                                continue

                            # check for contradictions, which should not be empty, since we are generating hard negatives
                            contradict_input = check_contradict(pos, new_neg)
                            check_args = llava_args.copy()
                            check_args['temperature'] = 0
                            check_args = DictObj(check_args)
                            # check_args['max_new_tokens'] = 100
                            full_contradict_prompt, conv = get_full_prompt(conv_mode=check_args.conv_mode,
                                                                            qs=contradict_input,
                                                                            mm_use_im_start_end=model.config.mm_use_im_start_end)
                            contradict_text = llava_generate(full_contradict_prompt, conv, image_tensor, tokenizer, check_args, model).strip()

                            # contradict_text = check_contradict(pos, new_neg, check_params, args.llm, client)
                            print('contradict response: ', contradict_text)

                            contradict_dict = json.loads(contradict_text)

                            if not contradict_dict['contradictions']:
                                attempts += 1
                                print('number of futile attempts: ', attempts)
                                if attempts == 5:
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

                        gen_prompts.append(new_neg.lower())
                        gen_changes.append(text_change)
                        gen_contradicts.append(contradict_dict)

                    else:
                        raise ValueError("Error: generated result in None.")

                if not include:
                    break

            # if include:
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

            # fill in remaining ones with empty strings, which we will skip over during the evaluation/inference step
            for miss_ind in range(len(gen_prompts), args.num_responses):
                curr_dict[f'new_neg_{miss_ind}'] = ''
                curr_dict[f'new_change_{miss_ind}'] = ''
                curr_dict[f'new_contradict_{miss_ind}'] = ''

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