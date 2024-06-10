# torch imports
import torch
import torch.nn.functional as F

# misc imports
import numpy as np
from PIL import Image

# from dec_vl_eval.src.utils.utils_llava import compute_image_tensor_llava
# from dec_vl_eval.src.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
# from dec_vl_eval.src.llava.mm_utils import  tokenizer_image_token
# from dec_vl_eval.src.llava.conversation import conv_templates

from dec_vl_eval.src.utils.utils_minigpt4v2 import chat_encode_img


letters_2options = ['A. ', 'B. ']
answer_letters_2options = ['A', 'B']

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
    return prompt

def get_labels_for_llava(instruction_w_answer, instruction_only, ignore_index):
    #  #  only the response tokens are not -100, the rest (system_promt, image placeholder, instructions) are -100
    labels = instruction_w_answer.clone()
    labels[: len(instruction_only)] = ignore_index
    # labels[image_position] = image_token_index
    return labels

def score_options(
    model,
    model_name,
    vis_processor,
    tokenizer,
    sample,
    multiple_choices,
    correct_choice,
    args=None,
    # for llava
    conv_templates = None,
    # for minigpt4v2
    conv_vision_minigpt4v2 = None,
    chat_minigpt4v2 = None,
):

    # Samples is the datapoint of evaluation
    # proc_sample = {
    #     "image_path": file_name,
    #     "image": Image.open(file_name).convert("RGB"),
    #     "prompt": prompt,
    # }

    # ORIGINAL CODE
    # # scores = torch.zeros(len(multiple_choices))
    #
    # # Look at our one image.

    if 'llava-v1.5' in model_name or 'llava-v1.6' in args.model:
        image, image_tensor = compute_image_tensor_llava(sample["image_path"], args, model, vis_processor)
        # TODO   instruction should also contain the options of the answers
        #   currently this is not implementated
        # if 'ARO' in args.eval_dataset:
        #     extended_instruction = sample["prompt"] + '\nOptions: ' + ' '.join(
        #         [letters_2options[i] + multiple_choices[i] for i in range(len(multiple_choices))]
        #     ) + '\n' + "Answer:"
        # else:
        extended_instruction = (sample["prompt"] + '\n' + '\n'.join([letters_2options[i] + multiple_choices[i] for i in range(len(multiple_choices))])
                                + '\n' + "Answer with the option's letter from the given choices directly.")
        full_instruction_prompt = get_full_prompt(conv_mode=args.conv_mode, qs= extended_instruction, mm_use_im_start_end=model.config.mm_use_im_start_end)
        extended_responses = answer_letters_2options

    elif 'minigpt4v2' in model_name:
        img_embed_query = chat_encode_img(chat_minigpt4v2, sample["image_path"]) # (1, 256, 4096)

    elif "instruct" in model_name or "blip2" in model_name:
        img = vis_processor(images=sample["image"])  #  InstructBlipProcessor : Constructs an InstructBLIP processor which wraps a BLIP image processor and a LLaMa/T5 tokenizer into a single
        # instr = tokenizer(text=sample["prompt"])  #  InstructBlipProcessor : Constructs an InstructBLIP processor which wraps a BLIP image processor and a LLaMa/T5 tokenizer into a single

    elif 'internlm' in model_name:
        options_prompt = f'A. {multiple_choices[0]} B. {multiple_choices[1]} '
        img_prompt = '[UNUSED_TOKEN_146]user\n'
        context = 'N/A'
        options_prompt = options_prompt.strip()
        mid_prompt = 'Question: ' + sample['prompt'] + '\nContext: ' + context + '\nOptions: ' + options_prompt
        ans_prompt = '[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\nThe answer is'
        full_instruction_prompt = img_prompt + mid_prompt + ans_prompt

    scores = {}
    gen_text = {}
    with (torch.no_grad()):
        if 'llava-v1.5' in model_name or 'llava-v1.6' in args.model:
            input_ids_instruction = tokenizer_image_token(full_instruction_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')  # IMAGE_TOKEN_INDEX = -200,
        for option_id, answer_text in enumerate(multiple_choices):
            caps_ids = tokenizer(text=answer_text) #  length of 16        [/INST]
            if 'llava-v1.5' in model_name or 'llava-v1.6' in args.model:
                input_ids_choice = tokenizer(text= extended_responses[option_id] )['input_ids']
                input_ids_ = torch.cat((input_ids_instruction, torch.tensor(input_ids_choice[1:]))).to(device='cuda') # remove the start token of the choice
                labels = get_labels_for_llava(input_ids_, input_ids_instruction, IGNORE_INDEX)
                input_data = {
                    'images': image_tensor,
                    'input_ids': input_ids_[None, :],
                    # 'attention_mask': torch.tensor([ [1] * len(instr) ], device='cuda'),
                    'labels': labels[None, :] }
                out_dict = model(**input_data, return_dict=True)
            elif 'internlm' in model_name:
                input_text = full_instruction_prompt
                instr_only = tokenizer(text=input_text, )
                input_ids_instruction = torch.tensor(instr_only['input_ids'])

                instr = tokenizer(text=input_text + ' ' + answer_letters_2options[option_id])
                input_ids_ = torch.tensor(instr['input_ids'])

                labels = get_labels_for_llava(instruction_w_answer=input_ids_, instruction_only=input_ids_instruction,
                                              ignore_index=IGNORE_INDEX)

                # input_data = {
                #     'pixel_values': torch.tensor(np.array(img['pixel_values']), device='cuda'),  # (1, 3, 224, 224)
                #     'input_ids': input_ids_[None, :].to(device='cuda'),
                #     'labels': labels[None, :].to(device='cuda')
                # }

                pt1, need_bos = 0, True
                embeds = []
                im_mask = []
                images = [sample["image_path"]]
                images_loc = [0]
                for i, pts in enumerate(images_loc + [len(input_text)]):
                    subtext = input_text[pt1:pts]
                    if need_bos or len(subtext) > 0:
                        text_embeds = model.encode_text(subtext, add_special_tokens=need_bos)
                        embeds.append(text_embeds)
                        im_mask.append(torch.zeros(text_embeds.shape[:2]).cuda())
                        need_bos = False
                    if i < len(images):
                        image = Image.open(images[i]).convert('RGB')
                        image = model.vis_processor(image).unsqueeze(0).cuda()
                        image_embeds = model.encode_img(image)
                        embeds.append(image_embeds)
                        im_mask.append(torch.ones(image_embeds.shape[:2]).cuda())
                    pt1 = pts
                embeds = torch.cat(embeds, dim=1)
                im_mask = torch.cat(im_mask, dim=1)
                im_mask = im_mask.bool()

                out_dict = model(
                    inputs_embeds=embeds[None,:].to(device='cuda'), im_mask=im_mask[None,:].to(device='cuda'), labels=labels[None,:].to(device='cuda'), return_dict=True,
                    temperature=1.0, max_new_tokens=5, num_beams=5, do_sample=False, repetition_penalty=1.0
                )

            elif 'minigpt4v2' in model_name:
                CURRENT_CONVERSATION = conv_vision_minigpt4v2.copy()
                input_text = '<Img><ImageHere></Img>' + sample["prompt"] + ' ' + answer_text
                chat_minigpt4v2.ask( input_text, CURRENT_CONVERSATION )
                img_list = [img_embed_query]  #  # img embed query is  (1, 256, 4096)
                # answer_prepare -  prepare the configuration
                generation_dict = chat_minigpt4v2.answer_prepare(CURRENT_CONVERSATION, img_list=img_list)  #  concatenate the input for the input image and input text  <s>[INST] <Img> + 256 visual tokens + </Img> What is man doing in the image?
                inputs_embeds = generation_dict['inputs_embeds']  #  this includes both image tokens and text tokens
                seq_len = inputs_embeds.shape[1]  #   the text tokens are at last
                input_data = {
                    "attention_mask": torch.ones((1, seq_len), device="cuda") ,  # what is the correct attention, should we only attend to the text tokens???   .to(inputs_embeds.device),
                    "input_ids": None ,
                    "inputs_embeds":  inputs_embeds,   #  MiniGPT4 uses inputs_embeds instead of input_ids
                    "labels": torch.tensor([caps_ids['input_ids']], device='cuda'),  #  the start token should be removed
                    "output_attentions": False,
                    "output_hidden_states": False,
                    "past_key_values": None,
                    "position_ids": torch.arange(seq_len, device='cuda' ),
                    "reduction": 'mean',
                    "use_cache": True,
                }
                out_dict = model.llama_model.forward_for_perplexity(**input_data, return_dict=True)

            elif "instruct" in model_name or "blip2" in model_name:
                if 'flan_t5' in model_name:
                    # input_text = sample["prompt"]
                    # instr = tokenizer(text=input_text)
                    # input_data = {
                    #     'pixel_values': torch.tensor(np.array(img['pixel_values']), device='cuda'),  # (1, 3, 224, 224)
                    #     'input_ids': torch.tensor([instr['input_ids']], device='cuda'),  # question and answer
                    #     'attention_mask': torch.tensor([instr['attention_mask']], device='cuda'),
                    #     'labels': torch.tensor([caps_ids['input_ids']], device='cuda')
                    #     # only the answer, the first token (start token) should be removed
                    # }

                    # if 'ARO' in args.eval_dataset:
                    #     input_text = sample["prompt"] + '\nOptions: ' + ' '.join(
                    #         [letters_2options[i] + multiple_choices[i] for i in range(len(multiple_choices))]
                    #     ) + '\n' + "Answer:"
                    # else:
                    input_text = sample["prompt"] + '\n' + '\n'.join(
                        [letters_2options[i] + multiple_choices[i] for i in range(len(multiple_choices))]
                        ) + '\n' + "Answer with the option's letter from the given choices directly. Answer:"
                    instr = tokenizer(text=input_text)
                    answer_ = tokenizer(text=answer_letters_2options[option_id])
                    input_data = {
                        'pixel_values': torch.tensor(np.array(img['pixel_values']), device='cuda'),  # (1, 3, 224, 224)
                        'input_ids': torch.tensor([instr['input_ids']], device='cuda'),  # question and answer
                        'attention_mask': torch.tensor([instr['attention_mask']], device='cuda'),
                        'labels': torch.tensor([answer_['input_ids']], device='cuda')
                        # only the answer, the first token (start token) should be removed
                    }

                elif 'vicuna' in model_name:
                    # if 'ARO' in args.eval_dataset:
                    #     input_text = sample["prompt"] + '\nOptions: ' + ' '.join(
                    #         [letters_2options[i] + multiple_choices[i] for i in range(len(multiple_choices))]
                    #     ) + '\n' + "Answer:"
                    # else:
                    # input_text = sample["prompt"] + ' ' + answer_text
                    input_text = sample["prompt"] + '\n' + '\n'.join([letters_2options[i] + multiple_choices[i] for i in range(len(multiple_choices))]
                                                                     ) + '\n' + "Answer with the option's letter from the given choices directly. Answer:"
                    instr_only = tokenizer(text=input_text, )
                    input_ids_instruction = torch.tensor(instr_only['input_ids'])

                    instr = tokenizer(text = input_text + ' ' + answer_letters_2options[option_id])
                    input_ids_ = torch.tensor(instr['input_ids'])

                    # input_ids_choice = tokenizer(text=answer_letters_2options[option_id] )['input_ids']
                    # input_ids_ = torch.cat((torch.tensor(input_ids_instruction), torch.tensor(input_ids_choice[1:])))
                    labels = get_labels_for_llava( instruction_w_answer = input_ids_, instruction_only = input_ids_instruction, ignore_index = IGNORE_INDEX)
                    input_data = {
                        'pixel_values': torch.tensor(np.array(img['pixel_values']), device='cuda'),  # (1, 3, 224, 224)
                        'input_ids': input_ids_[None,:].to(device='cuda'),
                        'labels': labels[None,:].to(device='cuda')
                    }

                if "instruct" in model_name:
                    input_data['qformer_input_ids'] = torch.tensor([instr['qformer_input_ids']], device='cuda')
                    input_data['qformer_attention_mask'] = torch.tensor([instr['qformer_attention_mask']], device='cuda')
                elif "blip2" in model_name:
                    input_data['input_ids'] = torch.tensor([instr['input_ids']], device='cuda')
                    input_data['attention_mask'] = torch.tensor([instr['attention_mask']], device='cuda')

                out_dict = model(**input_data, return_dict=True)
            scores[answer_text] = -out_dict['loss'].to('cpu').numpy()

            # Also save the generated text just for debugging really.
            if 'llava-v1.5' in model_name or 'llava-v1.6' in args.model:
                logits = out_dict["logits"]
                preds = F.softmax(logits, dim=-1).argmax(dim=-1)
                generated_text = tokenizer.batch_decode( sequences=preds, skip_special_tokens=True)
            elif "instruct" in model_name or "blip2" in model_name:
                logits = out_dict["language_model_outputs"]["logits"]
                preds = F.softmax(logits, dim=-1).argmax(dim=-1)
                generated_text = tokenizer.tokenizer.batch_decode(sequences=preds,skip_special_tokens=True)
            elif "minigpt4v2" in model_name:
                logits = out_dict["logits"]
                preds = F.softmax(logits, dim=-1).argmax(dim=-1)
                generated_text = tokenizer.batch_decode( sequences=preds,skip_special_tokens=True)
            elif 'internlm' in model_name:
                logits = out_dict["logits"]
                preds = F.softmax(logits, dim=-1).argmax(dim=-1)
                generated_text = tokenizer.batch_decode( sequences=preds, skip_special_tokens=True)

            gen_text[answer_text] = generated_text

    choice = max(  scores, key=lambda key: scores[key])  #   compare the scores of the two captions
    hit = int(choice == correct_choice)

    return hit, scores, gen_text

