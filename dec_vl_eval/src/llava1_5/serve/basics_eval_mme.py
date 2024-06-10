
import sys
import os
sys.path.append("")  # last level
sys.path.append(os.path.abspath('..'))
# import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

# from PIL import Image

import requests
from PIL import Image
from io import BytesIO
# from transformers import TextStreamer
from utils_.utils import make_dir
import os.path as osp
from tqdm import tqdm

from transformers import TextStreamer

from cli import load_image

def get_MME_data_path_dict(args):
    # MME_dir = '/data1/lin/data/LMM_benchmarks/MME/MME_Benchmark_release_version'
    MME_dir = args.MME_dir

    MME_data_path_dict = {
        'artwork': osp.join(MME_dir, 'artwork/questions_answers_YN_merged'),
        'celebrity': osp.join(MME_dir, 'celebrity/questions_answers_YN_merged'),
        'code_reasoning': osp.join(MME_dir, 'code_reasoning'),
        'color': osp.join(MME_dir, 'color'),
        'commonsense_reasoning': osp.join(MME_dir, 'commonsense_reasoning'),
        'count': osp.join(MME_dir, 'count'),
        'existence': osp.join(MME_dir, 'existence'),
        'landmark': osp.join(MME_dir, 'landmark/questions_answers_YN_merged'),
        'numerical_calculation': osp.join(MME_dir, 'numerical_calculation'),
        'OCR': osp.join(MME_dir, 'OCR'),
        'position': osp.join(MME_dir, 'position'),
        'posters': osp.join(MME_dir, 'posters/questions_answers_YN_merged'),
        'scene': osp.join(MME_dir, 'scene/questions_answers_YN_merged'),
        'text_translation': osp.join(MME_dir, 'text_translation')}
    return MME_data_path_dict

str_to_remove = [ ' </s>', '\n', '</s>', '.']

def main_eval_mme(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    # conv = conv_templates[args.conv_mode].copy()
    # if "mpt" in model_name.lower():
    #     roles = ('user', 'assistant')
    # else:
    #     roles = conv.roles

    make_dir(args.eval_result_dir)
    MME_data_path_dict = get_MME_data_path_dict(args)
    category_name_list = list(MME_data_path_dict.keys())

    for category_id, category_name in tqdm(enumerate(category_name_list)):
        if category_id >= 0:
            data_dir = MME_data_path_dict[category_name]
            print(f"Evaluating for {category_name} ...")
            f_write = open(osp.join(args.eval_result_dir, f'{category_name}.txt'), 'w+')
            # img_list = glob.glob(osp.join(data_dir, f'*{args.MME_img_format}' ))
            # for img_path in img_list:
            lines = open(osp.join( args.result_template_dir, f'{category_name}.txt') ).readlines()
            n_lines = len(lines)
            assert n_lines % 2 == 0
            # for line in open(osp.join( args.result_template_dir, f'{category_name}.txt') ):
            for line_id in range(0, n_lines, 2):
                imgfile_w_ext = lines[line_id].strip('\n').split('\t')[0]
                img_path = osp.join( data_dir, imgfile_w_ext )

                # load image
                image = load_image(img_path)
                # Similar operation in model_worker.py
                image_tensor = process_images([image], image_processor, args)
                if type(image_tensor) is list:
                    image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
                else:
                    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

                # filename = img_path.split('/')[-1].split('.')[0]
                # textfile_path = osp.join(data_dir, f'{filename}.txt')
                # lines = open(textfile_path, 'r').readlines()
                # assert len(lines) == 2
                for idx in range(2): #  every image has two Yes/No questions
                    line = lines[ line_id + idx]
                    _, question_, answer_ = line.strip('\n').split('\t')

                    # initialize the conversation
                    conv = conv_templates[args.conv_mode].copy()
                    inp = question_
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

                    input_ids = tokenizer_image_token(conv_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                    # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

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

                    new_line = line.strip('\n') + '\t' + outputs + '\n'
                    f_write.write(new_line)

            f_write.close()

