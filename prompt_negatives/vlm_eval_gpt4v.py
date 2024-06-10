# from cvar_pyutils.debugging_tools import set_remote_debugger

# Our library imports
from args import get_args
from dec_vl_eval.src.utils.directory import option_prompt_dict, generic_prompt_dict
from dec_vl_eval.src.datasets import get_benchmark_dataset
from dec_vl_eval.src.models import get_model, get_caption_model, model_generate, model_add_captions, score_options, t2i_score

# llava dependencies
from dec_vl_eval.src.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from dec_vl_eval.src.llava.conversation import conv_templates, SeparatorStyle
from dec_vl_eval.src.llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from dec_vl_eval.src.utils.utils_llava import load_image, str_to_remove, compute_image_tensor_llava

# minigpt4v2 dependencies
# from dec_vl_eval.src.minigpt4.conversation.conversation import CONV_VISION_minigptv2, CONV_VISION_minigptv2_for_perplexity

# t2i_metric dependencies
# from dec_vl_eval.src.t2i_metrics import get_score_model

# openai imports
import base64
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Misc imports
import numpy as np
import os
import os.path as osp
import pandas as pd
from PIL import Image
import random
import torch
import time
import getpass
import json


load_dotenv()
API_KEY = os.getenv("OPENAI_KEY", None)
HEADERS = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {API_KEY}"
}

"""
ln -s llava1_5 llava
ln -s llava1_6 llava
"""
def save_df_results( pickle_dir,
    val_records
):
    if os.path.exists(pickle_dir):
    # Load the existing .pkl file
        old_result_df = pd.read_pickle(pickle_dir) 
    else:
        old_result_df = pd.DataFrame([])

    current_results_df = pd.DataFrame(val_records) 
    combined_df = pd.concat([old_result_df, current_results_df], axis=0)
    combined_df.to_pickle(pickle_dir)

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

# OpenAI API helper functions
# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def fetch(sample, multiple_choices):
    '''

    :param img_path:
    :param prompt:
    :return:
    '''
    messages = []
    letters_2options = ['A. ', 'B. ']

    input_text = sample["prompt"] + '\n' + '\n'.join(
            [letters_2options[i] + multiple_choices[i] for i in range(len(multiple_choices))]
        ) + '\n' + "Answer with the option's letter from the given choices directly. Answer:"
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"{input_text}"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(sample['image_path'])}"
                }
            },
        ]
    })

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": messages,
        "max_tokens": 128
    }

    while True:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=HEADERS, json=payload)
        if response.status_code == 200:
            break

    return response.json()

if __name__ == '__main__':
    # args
    args = get_args()
    args.username = getpass.getuser()
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
    # print(f'Setting seed to: {SEED}')
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    # Get the dataset  -   here the prompt template will only be used for "generate" inference, the prompt template for "loss_score" inference is hardcoded in the score_options function
    if args.prompt_type == 'option':
        if 'llava-v1.5' in args.model or 'llava-v1.6' in args.model:
            assert args.prompt == '11'
        else:
            assert args.prompt == '1'

        prompt_template = option_prompt_dict[args.prompt]
    elif args.prompt_type == 'generic':
        prompt_template = generic_prompt_dict[args.prompt]
    else:
        raise ValueError("Only two kinds of prompts, options or generics.")

    data_dict, options = get_benchmark_dataset(args, prompt_template)

    # Get the model
    if 'llava-v1.5' in args.model or 'llava-v1.6' in args.model:
        tokenizer, model, image_processor = get_model(args, device)  #   tokenizer: LlamaTokenizer,     model: LlavaLlamaForCausalLM,     image_processor: CLIPImageProcessor
    elif 'minigpt4v2' in args.model:
        chat_for_minigpt4v2, model, vis_processor = get_model(args, device)
        if args.select_by == "generate":
            conv_vision_minigpt4v2 = CONV_VISION_minigptv2
        elif args.select_by == "loss_score":
            conv_vision_minigpt4v2 = CONV_VISION_minigptv2_for_perplexity
    elif 'internlm' in args.model or  'idefics' in args.model:
        model, tokenizer = get_model(args, device)
    elif 'gpt' in args.model:
        model = None
    else:
        model, instructblip_combined_processors = get_model(args, device) # model: InstructBlipForConditionalGeneration,  vis_processors: InstructBlipProcessor

    if args.select_by == 't2i':
        if 'llava' in args.model:
            t2i_model = get_score_model(args.model)
        elif args.model == 'instructblip_flan_t5':
            t2i_model = get_score_model('instructblip-flant5-xl')
        else:
            print(f't2i_metrics not implemented yet for {args.model}.')

    # Keep a pandas dataframe which keeps track of all captions, answers, correctness, and arguments for evaluation
    if args.job_name is None:
        args.job_name = time.strftime("%Y%m%d_%H%M%S")
    result_dir = osp.join(args.eval_base_dir, args.experiment_name, args.job_name, args.select_by)
    # print('result_dir: ', result_dir)

    # If the args.result_dir does not exist, create it.
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    pickle_dir = os.path.join(result_dir, 'results_dataframe.pkl')

    # Save your results
    # results = {}
    base_record = {
        "model": args.model,
        "backend": args.backend,
        "prompt_id": args.prompt,
        "benchmark": args.eval_dataset,
        "select_by": args.select_by
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
        num_correct = 0
        all_samples = random.sample(data_dict[k], k=len(data_dict[k]))

        for item_id, sample in enumerate(all_samples):
            # each item in new_items has 0 the original sample index, 1 image path, 2 orig pos, and 3 orig neg,
            # 4 original gpt4v response, 5 q1 question, 6 q2 question, 7 q3 question

            # Get the filename, sometimes it doesn't exist.
            orig_id, file_name = sample[0], sample[1]
            if file_name is None or not os.path.exists(file_name):
                # print(f"File name for task {k}: {file_name} doesn't exist!")
                continue
            else:
                total_tested += 1
                # print(f'loaded dataset index {item_id}')
                # print(f'original index {orig_id}\n')

            orig_pos, orig_neg, prompt, correct = sample[2], sample[3], sample[5], sample[6]
            text_options = sample[-1]  # randomly shuffled
            neg, q_ind, n_ind = sample[7], sample[8], sample[9]
            generated_list = []

            # Samples is the datapoint of evaluation
            proc_sample = {
                "image_path": file_name,
                "image": Image.open(file_name).convert("RGB"),
                "prompt": prompt,
            }

            #   "generate" VQA inference mode
            if args.select_by == "generate":
                letters_2options = ['A. ', 'B. ']

                if 'llava-v1.5' in args.model or 'llava-v1.6' in args.model:
                    image, image_tensor = compute_image_tensor_llava(file_name, args, model, image_processor)

                    ####  Llava 1.5  "generate" inference
                    # initialize the conversation
                    extended_instruction = (prompt + '\n' + '\n'.join(
                        [letters_2options[i] + text_options[i] for i in range(len(text_options))])
                                            + '\n' + "Answer with the option's letter from the given choices directly.")
                    full_instruction_prompt, conv = get_full_prompt(conv_mode=args.conv_mode, qs=extended_instruction,
                                                                    mm_use_im_start_end=model.config.mm_use_im_start_end)

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
                    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
                    conv.messages[-1][-1] = outputs
                    for str_ in str_to_remove:
                        outputs = outputs.replace(str_, '')
                    generated_text = outputs

                elif 'minigpt4v2' in args.model:
                    raise NotImplementedError("Didn't implement the generation inference for Minigpt4.")

                elif 'gpt' in args.model:
                    error_count = 0
                    generated_text = None
                    while not generated_text:
                        try:
                            response_json = fetch(proc_sample, text_options)
                            for use_key, use_val in response_json['usage'].items():
                                base_record[use_key] = use_val
                            generated_text = response_json["choices"][0]["message"]["content"].strip().strip()

                        except json.decoder.JSONDecodeError:
                            error_count += 1
                            # print('Encountered json error in parsing model responses --> regenerating response')
                            # print('error count: ', error_count)
                            continue

                        except KeyError:
                            error_count += 1
                            # print('Key error --> regenerating response')
                            # print('error count: ', error_count)
                            continue

                        except Exception as e:
                            error_count += 1
                            # print(e)
                            # print('error count: ', error_count)
                            continue

                else:
                    ####  InstructBLIP, InternLM-XComposer, Idefics "generate" inference
                    generated_text = model_generate(
                        args,
                        model,
                        proc_sample,
                        instructblip_combined_processors if 'instructblip' in args.model else tokenizer,
                        generate_settings,
                        device,
                        text_options
                    )

                hit = int(generated_text[0].lower() == correct.lower()) if generated_text else 0
                if args.model == 'instructblip_vicuna_7b':
                    if len(generated_text[0]) > 1:
                        hit = int(generated_text.strip().lower() == correct.lower())
                num_correct += hit
                scores, scores_diff = None, None

            # perplexity inference mode
            elif args.select_by == 'loss_score':

                if 'llava-v1.5' in args.model or 'llava-v1.6' in args.model:

                    ####  Llava "loss_score" inference
                    hit, scores, generated_text = score_options(
                        model, # LlavaLlamaForCausalLM
                        args.model,
                        image_processor, # CLIPImageProcessor
                        tokenizer,  # LlamaTokenizer
                        proc_sample, # a dict of 3 keys: image_path, image, prompt
                        multiple_choices=text_options, # a list of the two captions
                        correct_choice=correct, # correct caption
                        args=args,
                        conv_templates=conv_templates
                    )
                elif 'internlm' in args.model:
                    hit, scores, generated_text = score_options(
                        model,  # InstructBlipForConditionalGeneration
                        args.model,
                        tokenizer,
                        tokenizer,
                        proc_sample,  # a dict of 3 keys: image_path, image, prompt
                        args=args,
                        multiple_choices=text_options,  # a list of the two captions
                        correct_choice=correct,  # correct caption
                    )
                elif 'minigpt4v2' in args.model:
                    #### Minigpt4v2  "loss_score" inference
                    hit, scores, generated_text = score_options(
                        model,  # MiniGPTv2
                        args.model,
                        vis_processor,  # CLIPImageProcessor
                        model.llama_tokenizer,  # LlamaTokenizer
                        proc_sample,  # a dict of 3 keys: image_path, image, prompt
                        multiple_choices=text_options,  # a list of the two captions
                        correct_choice=correct,  # correct caption
                        args=args,
                        conv_vision_minigpt4v2=conv_vision_minigpt4v2,
                        chat_minigpt4v2=chat_for_minigpt4v2
                    )
                else:
                    ####  InstructBLIP  "loss_score" inference
                    hit, scores, generated_text = score_options(
                        model, # InstructBlipForConditionalGeneration
                        args.model,
                        instructblip_combined_processors,  #  InstructBlipProcessor : Constructs an InstructBLIP processor which wraps a BLIP image processor and a LLaMa/T5 tokenizer into a single
                        instructblip_combined_processors,
                        proc_sample,   #   a dict of 3 keys: image_path, image, prompt
                        args=args,
                        multiple_choices=text_options,  #  a list of the two captions
                        correct_choice=correct,   # correct caption
                    )
                scores_diff = scores[neg] - scores[correct]
                num_correct += hit
            else:
                print(f"Inference mode {args.select_by} is not implemented.")

            # save current sample record
            item_record = {
                "id": item_id,
                'orig_id': orig_id,
                "class": k,
                "orig_pos": orig_pos,
                'orig_neg': orig_neg,
                "image": file_name,
                "step3_gpt4v_response": sample[4],
                "question": prompt,
                "text_options": text_options,
                "correct": correct,
                "q_ind": q_ind,
                "gen_text": generated_text,
                "n_ind": n_ind,
                "neg": neg,
                "hit": hit,
                "scores": scores,
                "scores_diff": scores_diff,
                "accuracy": hit,
                "mid_accuracy": float(num_correct) / float(total_tested)
            }

            record = {**item_record, **base_record}
            val_records.append(record)

            # if args.select_by == 'generate':
            #     print(f'Generated text: {generated_text}')
            # elif args.select_by == 'loss_score':
            #     print(f'Scores: {scores}')

            # print(f'Mid acc: {item_record["mid_accuracy"]}')
            # print('-' * 50)

            if total_tested % args.save_every == 0:   
                # print("Saving results!")
                # Write the answer to the record files
                # and reset the accumulator
                save_df_results(pickle_dir, val_records)
                val_records = []

            if args.num_samples is not None and total_tested >= args.num_samples:
                break

        # After task save as well
        save_df_results(pickle_dir, val_records)
        val_records = []
        
    # Save at the end of the task
    save_df_results(pickle_dir, val_records)
    print('Script finished.')