# from cvar_pyutils.debugging_tools import set_remote_debugger

# Our library imports
from args import get_args
from dec_vl_eval.src.utils.directory import option_prompt_dict, generic_prompt_dict
from dec_vl_eval.src.datasets import get_benchmark_dataset
from dec_vl_eval.src.models import get_model, get_caption_model, model_generate, model_add_captions, score_options

# llava dependencies
from dec_vl_eval.src.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from dec_vl_eval.src.llava.conversation import conv_templates, SeparatorStyle
from dec_vl_eval.src.llava.model.builder import load_pretrained_model
from dec_vl_eval.src.llava.utils import disable_torch_init
from dec_vl_eval.src.llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from dec_vl_eval.src.utils.utils_llava import load_image, str_to_remove, compute_image_tensor_llava

# Misc imports
import numpy as np
import os
import pandas as pd
from PIL import Image
import random
import torch
import time


def save_df_results(pickle_dir,
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


if __name__ == '__main__':
    # args
    args = get_args()

    # setup debug
    if args.debug:
        import pydevd_pycharm

        debug_ip = os.environ.get('SSH_CONNECTION', None)
        pydevd_pycharm.settrace(debug_ip, port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)
        # set_remote_debugger(None, args.debug_port)

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

    # LLaVA dependencies
    if 'llava-v1.5' in args.model or 'llava-v1.6' in args.model:
        # llava dependencies
        from dec_vl_eval.src.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
            DEFAULT_IM_END_TOKEN
        from dec_vl_eval.src.llava.conversation import conv_templates, SeparatorStyle
        from dec_vl_eval.src.llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, \
            KeywordsStoppingCriteria
    elif 'llava-v1.6' in args.model:
        from dec_vl_eval.src.llava1_6.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
            DEFAULT_IM_END_TOKEN
        from dec_vl_eval.src.llava1_6.conversation import conv_templates, SeparatorStyle
        from dec_vl_eval.src.llava1_6.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, \
            KeywordsStoppingCriteria

    # Get the dataset
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
    else:
        model, instructblip_combined_processors = get_model(args, device) # model: InstructBlipForConditionalGeneration,  vis_processors: InstructBlipProcessor

    caption_model = None

    # Add potentially an auxiliary model for injection.
    if args.caption_dataset is not None:
        caption_model = get_caption_model(args, model, device)

    # Keep a pandas dataframe which keeps track of all captions, answers, correctness, and arguments for evaluation
    # TODO: Define in the args file the output dir
    if args.job_name is None:
        args.job_name = time.strftime("%Y%m%d_%H%M%S")
    result_dir = f'{args.eval_base_dir}/{args.experiment_name}/{args.job_name}'
    print('result_dir: ', result_dir)

    # If the args.result_dir does not exist, create it.
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    pickle_dir = os.path.join(result_dir, 'results_dataframe.pkl')
    f_write = open(os.path.join(result_dir, 'results.txt'), 'w+')

    # Save your results
    results = {}
    # TODO: Save the arguments file and log them.
    base_record = {
        "model": args.model,
        "backend": args.backend,
        "prompt_id": args.prompt,
        "benchmark": args.eval_dataset,
        "select_by": args.select_by,
        "caption_dataset": args.caption_dataset,
        "pretrained_root": args.pretrained_root,
    }

    f_write.write(str(base_record) + '\n')
    f_write.flush()

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

        num_correct = 0
        total_tested = 0

        for item_id, sample in enumerate(data_dict[k]):

            # Get the filename, sometimes it doesn't exist.
            file_name = sample[0]
            if file_name is None or not os.path.exists(file_name):
                print(f"File name for task {k}: {file_name} doesn't exist!")
                continue
            else:
                total_tested += 1

            item_record = {
                "class": k,
                "id": item_id,
                "image": file_name
            }

            curr_correct = 0
            # get scores for each of the selected top k LLM-generated negatives
            for neg_select_ind in range(args.num_neg_options):
                # Construct the correct prompt
                prompt = sample[1 + (3 * neg_select_ind)]

                # Get the groundtruth
                gt = sample[2 + (3 * neg_select_ind)]

                # Text options
                text_options = sample[3 + (3 * neg_select_ind)]
                neg = [alt for alt in text_options if alt != gt][0]

                # Samples is the datapoint of evaluation
                proc_sample = {
                    "image_path": file_name,
                    "image": Image.open(file_name).convert("RGB"),
                    "prompt": prompt,
                }

                if args.caption_dataset is not None:
                    proc_sample = model_add_captions(
                        args,
                        caption_model,
                        proc_sample,
                        instructblip_combined_processors,
                        device
                    )
                    prompt = proc_sample["prompt"]

                # "generate" VQA inference mode
                if args.select_by == "generate":
                    # generated_text = model_generate(
                    #     args,
                    #     model,
                    #     proc_sample,
                    #     vis_processors,
                    #     generate_settings,
                    #     device
                    # )
                    # # See whether the first letter is correct
                    # scores = None
                    # scores_diff = None

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
                    scores_diff = None

                    if args.eval_dataset == 'SEED':
                        generated_char = generated_text[1] if len(generated_text) > 1 else generated_text[0]
                        hit = int(generated_char.lower() == gt.lower())
                    else:
                        hit = int(generated_text.lower() == gt.lower())

                # "perplexity" inference mode
                elif args.select_by == "loss_score":
                    # hit, scores, generated_text = score_options(
                    #     model,
                    #     args.model,
                    #     vis_processors,
                    #     proc_sample,
                    #     multiple_choices=text_options,
                    #     correct_choice=gt,
                    # )
                    # scores_diff = scores[neg] - scores[gt]

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
                    scores_diff = scores[neg] - scores[gt]

                else:
                    raise NotImplementedError("Didn't implement this kind of generation.")

                item_record.update({
                    f'accuracy_{neg_select_ind}': hit,
                    f'correct_option_{neg_select_ind}': gt,
                    f'prompt_{neg_select_ind}': prompt,
                    f'gen_text_{neg_select_ind}': generated_text,
                    f'scores_{neg_select_ind}': scores,
                    f'scores_diff_{neg_select_ind}': scores_diff
                })

                if hit:
                    curr_correct = 1

            num_correct += curr_correct
            results[k] = float(num_correct) / float(total_tested)

            item_record.update({
                "mid_accuracy": results[k],
            })
            if args.eval_dataset == 'LLM':
                item_record.update({'original_id': sample[-1]})

            record = {**item_record, **base_record}
            val_records.append(record)

            # print(f'Prompt: {prompt}\n')
            # print(f'Options Given: {text_options}\n')
            # print(f'Generated Text: {generated_text}\n')
            #
            # if args.select_by == "loss_score":
            #     print(f"Scores: {scores}\n")
            #
            # print(f'Correct Option: {gt}')
            print(f'Item id: {item_id}\n')
            print(f'Cumulative Accuracy: {results}')

            if curr_correct != 1:
                print("INCORRECT!")

            print('-' * 50)

            if total_tested % args.save_every == 0:
                print("Saving results!")
                # Write the answer to the record files
                # and reset the accumulator
                save_df_results(pickle_dir, val_records)
                val_records = []

            if args.num_samples is not None and total_tested > args.num_samples:
                break

        # After task save as well
        save_df_results(pickle_dir, val_records)
        val_records = []

    # Save at the end of the task
    save_df_results(pickle_dir, val_records)

    print("Final Results:")
    for key, item in results.items():
        print(key, item)
        f_write.write(f'{key} {item:.4f}\n')

    f_write.close()
