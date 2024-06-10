import time
import json
import os
import sys
import random
import re
import string
import argparse
from functools import partial
from PIL import Image

# import fire
import tqdm
import torch
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

# from dotenv import load_dotenv
# from genai.credentials import Credentials
# from genai.model import Model
# from genai.schemas import GenerateParams
# from genai.schemas.generate_params import LengthPenalty

from dec_vl_eval.src.utils.directory import option_prompt_dict
from dec_vl_eval.src.datasets.benchmark_datasets import get_benchmark_dataset
from dec_vl_eval.src.models import get_model, get_caption_model, model_generate, model_add_captions, score_options, t2i_score

# llava dependencies
from dec_vl_eval.src.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from dec_vl_eval.src.llava.conversation import conv_templates, SeparatorStyle
# from dec_vl_eval.src.llava.model.builder import load_pretrained_model
# from dec_vl_eval.src.llava.utils import disable_torch_init
from dec_vl_eval.src.llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from dec_vl_eval.src.utils.utils_llava import load_image, str_to_remove, compute_image_tensor_llava

# t2i_metric dependencies
from dec_vl_eval.src.t2i_metrics import get_score_model

# load_dotenv()
# API_KEY = os.getenv("GENAI_KEY", None)
# API_URL = os.getenv("GENAI_API", None)

def construct_prompt(
        question,
        image_path,
        pos,
        neg,
        select_by
):
    options = ['A', 'B']

    opt_txt = np.array([pos, neg])
    ixx = np.arange(len(opt_txt))
    np.random.shuffle(ixx)
    gt = np.where(ixx == 0)[0][0]
    txt = opt_txt[ixx]

    prompt = question
    if select_by == "generate":
        # prompt = f"Option A: <txt1>. Option B: <txt2>. {question.strip('?.,!;:')}, A or B?".replace('<txt1>', txt[0].strip('.')).replace('<txt2>', txt[1].strip('.'))
        gt = options[gt]
    elif select_by == "loss_score" or select_by == 't2i':
        # prompt = question
        gt = txt[gt]
    else:
        raise NotImplementedError("Selection method not implemented.")

    return {'image_path': image_path, "image": Image.open(image_path).convert("RGB"), 'prompt': prompt, 'correct': gt, 'pos': pos, 'neg': neg, 'options': txt}

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

def eval_generate(
        model,
        args,
        proc_sample,
        device
):
    image_path, prompt, correct, pos, neg, options = proc_sample['image_path'], proc_sample['prompt'], proc_sample['correct'], proc_sample['pos'], proc_sample['neg'], proc_sample['options']

    # generate evaluation method
    if args.select_by == 'generate':
        letters_2options = ['A. ', 'B. ']

        ####  Llava 1.5 or 1.6  "generate" inference
        if 'llava-v1.5' in args.model or 'llava-v1.6' in args.model:
            image, image_tensor = compute_image_tensor_llava(image_path, args, model, args.image_processor)
            ####  Llava 1.5  "generate" inference
            # initialize the conversation
            extended_instruction = (prompt + '\n' + '\n'.join(
                [letters_2options[i] + options[i] for i in range(len(options))])
                                    + '\n' + "Answer with the option's letter from the given choices directly.")
            full_instruction_prompt, conv = get_full_prompt(conv_mode=args.conv_mode, qs=extended_instruction,
                                                            mm_use_im_start_end=model.config.mm_use_im_start_end)

            # conv = conv_templates[args.conv_mode].copy()
            # inp = prompt
            # if image is not None:
            #     # first message
            #     if model.config.mm_use_im_start_end:
            #         inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            #     else:
            #         inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            #     conv.append_message(conv.roles[0], inp)
            #     image = None
            # else:
            #     # later messages
            #     conv.append_message(conv.roles[0], inp)
            # conv.append_message(conv.roles[1], None)
            # conv_prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(full_instruction_prompt, args.tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, args.tokenizer, input_ids)
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
                    stopping_criteria=[stopping_criteria])
            outputs = args.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            conv.messages[-1][-1] = outputs
            for str_ in str_to_remove:
                outputs = outputs.replace(str_, '')
            generated_text = outputs

        ####  InstructBLIP models "generate" inference
        else:
            generate_settings = {
                "length_penalty": 1.0,
                "repetition_penalty": 1.0,
                "num_beams": 5,
                # "max_length": 10,
                "min_length": 1,
                "max_new_tokens": 10,
                "top_p": 0.9,
                "temperature": 1.0,
                "do_sample": True
            }

            generated_text = model_generate(
                args,
                model,
                proc_sample,
                args.instructblip_combined_processors,
                generate_settings,
                device,
                options
            )
        # See whether the first letter is correct
        scores = None
        scores_diff = None

        hit = int(generated_text[0].lower() == correct.lower())

    #  "perplexity" inference mode
    elif args.select_by == "loss_score":
        if 'llava-v1.5' in args.model or 'llava-v1.6' in args.model:
            ####  Llava 1.5 or 1.6  "loss_score" inference
            hit, scores, generated_text = score_options(
                model,  # LlavaLlamaForCausalLM
                args.model,
                args.image_processor,  # CLIPImageProcessor
                args.tokenizer,  # LlamaTokenizer
                proc_sample,  # a dict of 3 keys: image_path, image, prompt
                multiple_choices=options,  # a list of the two captions
                correct_choice=pos,  # correct caption
                args=args,
                conv_templates=conv_templates
            )

        else:
            ####  InstructBLIP  "loss_score" inference
            hit, scores, generated_text = score_options(
                model,  # InstructBlipForConditionalGeneration
                args.model,
                args.instructblip_combined_processors, # InstructBlipProcessor : Constructs an InstructBLIP processor which wraps a BLIP image processor and a LLaMa/T5 tokenizer into a single
                args.instructblip_combined_processors,
                proc_sample,  # a dict including keys for: image_path, image, prompt
                multiple_choices=options,  # a list of the two captions
                correct_choice=pos,  # correct caption
            )

        scores_diff = scores[neg] - scores[pos]

    elif args.select_by == "t2i":
        hit, scores = t2i_score(proc_sample, options, pos, args.t2i_model)
        scores_diff = scores[neg] - scores[gt]

    return hit, scores, scores_diff


class DictObj:
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
               setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
               setattr(self, key, DictObj(val) if isinstance(val, dict) else val)

def prompt(
    output_path: str,
):

    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument("--eval_dataset",
                        default='SAMPLE_NEG_UPDATED',
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
    parser.add_argument("--select_by",
                        default='loss_score',
                        type=str,
                        help="Which evaluation method to use, for formatting in this case")

    # evaluation vlm model
    parser.add_argument("--eval_vlm",
                        default='instructblip_flan_t5',
                        type=str,
                        help="Which vlm to use for evaluation.")
    parser.add_argument("--prompt_model",
                        type=str,
                        default='llm',
                        help="Which type of model was used for prompting, either llm or vlm.")

    # output directory
    parser.add_argument("--output_path",
                        default=output_path,
                        type=str,
                        help="Path to output directory.")
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
        print(f"output dir: {args.output_path}")

    # get requested dataset by using helper function from benchmarks
    data_dict, options = get_benchmark_dataset(args=args)

    # getting VLM models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    methods = ['generate', 'loss_score']
    # methods = ['loss_score']
    # methods = ['t2i']

    # llava-1.5
    if args.eval_vlm == 'llava-v1.5-7b':
        llava_args = {
            'model': 'llava-v1.5-7b',
            'model_path': "liuhaotian/llava-v1.5-7b",
            # 'model_path': "/dccstor/leonidka1/irenespace/data/llava_weights/llava-v1.5-7b",
            'model_base': None,
            'load_8bit': False,
            'load_4bit': False,
            'prompt': 11,
            'temperature': 0,
            'max_new_tokens': 128,
            'conv_mode': "llava_v1",
            'select_by': None,
            'tokenizer': None,
            'image_processor': None,
            'device': device,
            't2i_model': get_score_model('llava-v1.5-7b')
        }
        llava_tokenizer, llava_model, llava_image_processor = get_model(DictObj(llava_args), device)  # llava-1.5

    elif args.eval_vlm == 'llava-v1.6-7b':
        llava_args = {
            'model': 'llava-v1.6-7b',
            'model_path': "liuhaotian/llava-v1.6-vicuna-7b",
            'model_base': None,
            'load_8bit': False,
            'load_4bit': False,
            'prompt': 11,
            'temperature': 0,
            'max_new_tokens': 128,
            'conv_mode': "llava_v1",
            'select_by': None,
            'tokenizer': None,
            'image_processor': None,
            'device': device,
            # 't2i_model': get_score_model('llava-v1.6-7b')
        }
        llava_tokenizer, llava_model, llava_image_processor = get_model(DictObj(llava_args), device)  # llava-1.6

    elif 'instructblip' in args.eval_vlm:
        vlm_args = {
            'backend': 'hf',
            'model': args.eval_vlm
        }
        if args.eval_vlm == 'instructblip_flant5':
            vlm_args['t2i_model'] = get_score_model('instructblip-flant5-xl')

        vlm_model, vlm_combined_processors = get_model(DictObj(vlm_args), device)

    else:
        raise ValueError(f'No implementation for {args.eval_vlm} found.')

    # seed
    max_int_32bit = 2 ** 32 - 1
    SEED = int(round(time.time() * 1000)) % max_int_32bit
    print(f'Setting seed to: {SEED}')
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    for key, samples in data_dict.items():

        all_samples = random.sample(samples, k=len(samples))

        if args.dataset_partition and key != args.dataset_partition:
            continue

        for ind, sample in enumerate(all_samples):
            if args.num_samples and ind == args.num_samples:
                break

            save_fn = f"{args.output_path}/{args.eval_vlm}_sample-{ind:08d}.csv"
            if os.path.exists(save_fn):
                continue
            else:
                # first create empty csv to use filename for minimizing duplicates when running runme_prompt
                pd.DataFrame([]).to_csv(save_fn)

            print('-' * 66)
            print(f'Seeded sample index {ind}')
            print(f'Dataset item index {sample[0]}\n')

            pos = sample[2]
            print('original positive: ', pos)

            orig_neg = sample[3]
            print('original negative: ', orig_neg)

            new_neg = sample[6]
            print('new negative: ', new_neg)

            new_change = sample[7]
            print('new change: ', new_change)

            contradict_check = sample[8]
            print('contradiction check: ', contradict_check)

            question = sample[4]
            print('prompt: ', question + '\n')

            # make sure image path exists
            image_path = sample[1]
            if not os.path.exists(image_path):
                continue
            print('image path: ', image_path)

            include = True
            selected_reqs = None

            # for saving file
            curr_dict = {
                "orig_id": sample[0],  # original index from the loaded dataset
                'image': image_path,  # original image path
                "class": key,
                'orig_pos': pos,  # original positive
                'orig_neg': orig_neg,  # original negative,
                'new_neg': new_neg,
                'new_change': new_change,
                'contradiction_check': contradict_check
            }

            # now, we check the evaluation method requirements on the VLMs
            reqs = {f'{args.eval_vlm}_pass': True}

            if args.eval_vlm in {'llava-v1.5-7b', 'llava-v1.6-7b'}:
                for method in methods:
                    proc_sample = construct_prompt(
                        question,
                        image_path,
                        pos,
                        new_neg,
                        method
                    )
                    # print(f'{method} prompt: ' + proc_sample['prompt'])
                    llava_args.update(
                        {'select_by': method, 'tokenizer': llava_tokenizer, 'image_processor': llava_image_processor})

                    hit, scores, scores_diff = eval_generate(llava_model, DictObj(llava_args), proc_sample, device)
                    reqs[f'llava_{method}'] = {'hit': hit, 'scores': scores, 'scores_diff': scores_diff, 'prompt': proc_sample['prompt']}
                    if hit:  # this means the VLM correctly chooses the ground truth caption i.e. negative is not hard enough
                        reqs[f'{args.eval_vlm}_pass'] = False

            elif 'instructblip' in args.eval_vlm:
                # instructblip models
                for method in methods:
                    proc_sample = construct_prompt(
                        question,
                        image_path,
                        pos,
                        new_neg,
                        method
                    )
                    # print(f'{method} prompt: ' + proc_sample['prompt'])
                    vlm_args.update(
                        {'select_by': method, 'instructblip_combined_processors': vlm_combined_processors})

                    hit, scores, scores_diff = eval_generate(vlm_model, DictObj(vlm_args), proc_sample,
                                                             device)
                    reqs[f'{args.eval_vlm}_{method}'] = {'hit': hit, 'scores': scores, 'scores_diff': scores_diff, 'prompt': proc_sample['prompt']}
                    if hit:  # this means the VLM correctly chooses the ground truth caption i.e. negative is not hard enough
                        reqs[f'{args.eval_vlm}_pass'] = False

            else:
                raise ValueError(f'No implementation for {args.eval_vlm} found.')

            # if not reqs[f'{args.eval_vlm}_{curr_choice}']['hit']:
            if reqs[f'{args.eval_vlm}_pass']:
                selected_reqs = reqs
                # print(f'\nDataset item {sample[-1]} passes.')
                # print('Reqs: ', reqs)

            else:
                include = False
                # print(f'\nDataset item {sample[-1]} does not pass.')
                # print('Reqs: ', reqs)

            if include:
                curr_dict['select_reqs'] = selected_reqs
                pd.DataFrame([curr_dict]).to_csv(save_fn)
                # print(f'finished seeded sample {ind}: saved filename for item {sample[-1]} at {save_fn}')

    return

if __name__ == '__main__':

    # output directory
    job = time.strftime("%Y%m%d_%H%M")
    OUTPUT_DIR = f"/dccstor/leonidka1/irenespace/prompt_negatives_revised/llava_mix/{job}"
    prompt(output_path=OUTPUT_DIR)
    print('Script finished.')
