import time
import json
import os
import sys
import random
import re
import string
import argparse
from functools import partial
import numpy as np
import pandas as pd
from PIL import Image
import torch

from args import get_args
from dec_vl_eval.src.utils import directory
from dec_vl_eval.src.datasets import get_benchmark_dataset
from dec_vl_eval.src.models import get_model, get_caption_model, model_generate, model_add_captions, score_options

# llava dependencies
from dec_vl_eval.src.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from dec_vl_eval.src.llava.conversation import conv_templates, SeparatorStyle
# from dec_vl_eval.src.llava.model.builder import load_pretrained_model
# from dec_vl_eval.src.llava.utils import disable_torch_init
from dec_vl_eval.src.llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from dec_vl_eval.src.utils.utils_llava import load_image, str_to_remove, compute_image_tensor_llava


def save_df_results(csv_path,
                    val_records
                    ):
    if os.path.exists(csv_path):
        # Load the existing .csv file
        old_result_df = pd.read_csv(csv_path)
    else:
        old_result_df = pd.DataFrame([])

    current_results_df = pd.DataFrame(val_records)
    combined_df = pd.concat([old_result_df, current_results_df], ignore_index=True)
    combined_df.to_csv(csv_path)

if __name__ == '__main__':

    # args
    args = get_args()

    # # first, concatenate all generated samples into one updated csv file
    # FOLDER='/dccstor/leonidka1/irenespace/scratch/prompt_negatives/aro/20231112_1356'
    #
    # dfs = []
    # for csv_path in sorted(os.listdir(FOLDER)):
    #     # consider subset of 100 for now first
    #     # if int(csv_path[-8:-4]) >= 100:
    #     #     break
    #
    #     if csv_path != f'falcon-180b_{args.dataset_partition}_all-samples.csv':
    #         dfs.append(pd.read_csv(f'{FOLDER}/{csv_path}'))
    #
    # df_all = pd.concat(dfs, ignore_index=True)
    # df_all.to_csv(f'{FOLDER}/falcon-180b_{args.dataset_partition}_all-samples.csv')

    # setup debug
    if args.debug:
        import pydevd_pycharm

        debug_ip = os.environ.get('SSH_CONNECTION', None)
        pydevd_pycharm.settrace(debug_ip, port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device:
        device = torch.device(args.device)

    # seed
    max_int_32bit = 2 ** 32 - 1
    SEED = int(round(time.time() * 1000)) % max_int_32bit
    print(f'Setting seed to: {SEED}')
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    # Get the dataset
    if args.prompt_type == 'option':
        if 'llava-v1.5' in args.model or 'llava-v1.6' in args.model:
            assert args.prompt == '11'
        else:
            assert args.prompt == '1'

        prompt_template = directory.option_prompt_dict[args.prompt]
    elif args.prompt_type == 'generic':
        prompt_template = directory.generic_prompt_dict[args.prompt]
    else:
        raise ValueError("Only two kinds of prompts, options or generics.")

    data_dict, options = get_benchmark_dataset(args, prompt_template)

    # Get the model
    if 'llava-v1.5' in args.model or 'llava-v1.6' in args.model:
        tokenizer, model, image_processor = get_model(args,
                                                      device)  # tokenizer: LlamaTokenizer,     model: LlavaLlamaForCausalLM,     image_processor: CLIPImageProcessor
    else:
        model, instructblip_combined_processors = get_model(args,
                                                            device)  # model: InstructBlipForConditionalGeneration,  vis_processors: InstructBlipProcessor
    # Keep a pandas dataframe which keeps track of all captions, answers, correctness, and arguments for evaluation
    if args.job_name is None:
        args.job_name = f'{args.dataset_partition}_{args.model}'
    result_dir = f'/dccstor/leonidka1/irenespace/scratch/prompt_negatives/aro/results/{args.job_name}'
    print('result_dir: ', result_dir)

    # If the args.result_dir does not exist, create it.
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    csv_path = os.path.join(result_dir, 'score_diffs.csv')

    # Save your results
    # results = {}
    base_record = {
        "model": args.model,
        "backend": args.backend,
        "prompt_id": args.prompt,
        "benchmark": args.eval_dataset,
        "select_by": args.select_by,
        "pretrained_root": args.pretrained_root,
    }

    # Generating params
    generate_settings = {
        "length_penalty": 1.0,
        "repetition_penalty": 1.0,
        "num_beams": args.num_beams,
        # "max_length": 10,
        "min_length": 1,
        "max_new_tokens": 10,
        "top_p": 0.9,
        "temperature": 1.0,
    }

    val_records = []

    for k in data_dict.keys():

        total_tested = 0

        for item_id, sample in enumerate(data_dict[k]):

            # Get the filename, sometimes it doesn't exist.
            file_name = sample[1]
            if file_name is None or not os.path.exists(file_name):
                print(f"File name for task {k}: {file_name} doesn't exist!")
                continue
            else:
                total_tested += 1

            print(f'Processing sample item {item_id}\n\n')

            item_record = {
                "orig_id": sample[0],
                "image": file_name,
                "class": k,
                "orig_pos": sample[2],
                "orig_neg": sample[3]
            }

            for gen_ind in range(5):
                adj_ind = 4 + (3 * gen_ind)

                # Construct the correct prompt
                prompt = sample[adj_ind]

                # Get the groundtruth
                gt = sample[adj_ind + 1]

                # Text options
                text_options = sample[adj_ind + 2]
                neg = [alt for alt in text_options if alt != gt][0]

                # Samples is the datapoint of evaluation
                proc_sample = {
                    "image_path": file_name,
                    "image": Image.open(file_name).convert("RGB"),
                    "prompt": prompt,
                }

                # generation inference evaluation method
                if args.select_by == "generate":
                    if 'llava-v1.5' in args.model or 'llava-v1.6' in args.model:
                        image, image_tensor = compute_image_tensor_llava(file_name, args, model, image_processor)
                        ####  Llava 1.5  "generate" inference
                        # initialize the conversation
                        conv = conv_templates[args.conv_mode].copy()
                        inp = prompt
                        if image is not None:
                            # first message
                            if model.config.mm_use_im_start_end:
                                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                            else:
                                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                            conv.append_message(conv.roles[0], inp)
                            image = None
                        else:
                            # later messages
                            conv.append_message(conv.roles[0], inp)
                        conv.append_message(conv.roles[1], None)
                        conv_prompt = conv.get_prompt()

                        input_ids = tokenizer_image_token(conv_prompt, tokenizer, IMAGE_TOKEN_INDEX,
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
                                stopping_criteria=[stopping_criteria])
                        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
                        conv.messages[-1][-1] = outputs
                        for str_ in str_to_remove:
                            outputs = outputs.replace(str_, '')
                        generated_text = outputs


                    else:
                        ####  InstructBLIP  "generate" inference
                        generated_text = model_generate(
                            args,
                            model,
                            proc_sample,
                            instructblip_combined_processors,
                            generate_settings,
                            device
                        )
                    # See whether the first letter is correct
                    scores = None

                    hit = int(generated_text.lower() == gt.lower())

                # perplexity inference evaluation method
                elif args.select_by == "loss_score":
                    if 'llava-v1.5' in args.model or 'llava-v1.6' in args.model:
                        ####  Llava 1.5  "loss_score" inference
                        hit, scores, generated_text = score_options(
                            model,  # LlavaLlamaForCausalLM
                            args.model,
                            image_processor,  # CLIPImageProcessor
                            tokenizer,  # LlamaTokenizer
                            proc_sample,  # a dict of 3 keys: image_path, image, prompt
                            multiple_choices=text_options,  # a list of the two captions
                            correct_choice=gt,  # correct caption
                            args=args,
                            conv_templates=conv_templates
                        )
                    else:
                        ####  InstructBLIP  "loss_score" inference
                        hit, scores, generated_text = score_options(
                            model,  # InstructBlipForConditionalGeneration
                            args.model,
                            instructblip_combined_processors,
                            # InstructBlipProcessor : Constructs an InstructBLIP processor which wraps a BLIP image processor and a LLaMa/T5 tokenizer into a single
                            instructblip_combined_processors,
                            proc_sample,  # a dict of 3 keys: image_path, image, prompt
                            multiple_choices=text_options,  # a list of the two captions
                            correct_choice=gt,  # correct caption
                        )
                else:
                    raise NotImplementedError("Didn't implement this kind of generation.")


                if args.select_by == "loss_score":
                    diff = scores[neg] - scores[gt] # scores is a dictionary mapping A and B to floats

                item_record.update({
                    f"acc_{gen_ind}": hit,
                    f"new_neg_{gen_ind}": neg,
                    f"prompt_{gen_ind}": prompt,
                    f"gen_text_{gen_ind}": generated_text,
                    f"scores_{gen_ind}": scores,
                    f"diff_scores_{gen_ind}": diff if args.select_by == "loss_score" else None
                })


                # print(f'Prompt {gen_ind}: {prompt}\n')
                # print(f'Options Given {gen_ind}: {text_options}\n')
                # print(f'Generated Text {gen_ind}: {generated_text}\n')

                if args.select_by == "loss_score":
                    print(f"Scores {gen_ind}: {scores}\n")
                    print(f'Selected Option {gen_ind}: {max(scores, key=lambda opt: scores[opt])}\n')
                    print(f"Scores difference {gen_ind}: {diff}\n")

                # print(f'Correct Option {gen_ind}: {gt}')
                # print(f'Cumulative Accuracy: {results}')

                if hit != 1:
                    print("INCORRECT!")

            print('-' * 50)

            record = {**item_record, **base_record}
            val_records.append(record)

            if total_tested % args.save_every == 0:
                print("Saving results!")
                # Write the answer to the record files
                # and reset the accumulator
                save_df_results(csv_path, val_records)
                val_records = []

            if args.num_samples is not None and total_tested > args.num_samples:
                break


        # After task save as well
        save_df_results(csv_path, val_records)
        val_records = []

    # Save at the end of the task
    save_df_results(csv_path, val_records)
    print('results saved to csv at: ', csv_path)
