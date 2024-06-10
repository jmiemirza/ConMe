# Local imports
from .vlc_data import VLC, VLC_ROOT
from .aro_datasets import VG_Relation, VG_Attribution, COCO_Order, Flickr30k_Order
from .seed_dataset import get_seed_image_data


# Misc imports
import numpy as np
from tqdm import tqdm
import itertools
import ast
from torch.utils.data import Dataset
import pandas as pd
import logging
import os
import json
import glob
import time
import re

class BaseCsvDataset(Dataset):

    def __init__(self, args, input_filename, prompt_template):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename)
         
        self.args = args
        self.images = df["image_id"].tolist()
        self.captions = df["caption"].tolist()
        self.hard_negs = [ast.literal_eval(ls_str) for ls_str in df["hard_negs"]]
        self.prompt_template = prompt_template
        self.select_by = args.select_by 

    def __getitem__(self, idx):

        POS = self.captions[idx]
        neg_list = list(self.hard_negs[idx])
        NEG_LIST = list(np.random.choice(neg_list, size=self.args.num_neg_options, replace=False))
        NUM_OPTIONS = len(NEG_LIST) + 1
        options = [chr(i) for i in range(65, 65 + NUM_OPTIONS)]

        opt_txt = np.array([POS] + NEG_LIST)
        ixx = np.arange(len(opt_txt))
        np.random.shuffle(ixx)
        gt = np.where(ixx == 0)[0][0]
        txt = opt_txt[ixx]
 
        prompt = " ".join([f"Option {chr(i)}: <txt{i - 65}>" for i in range(65, 65 + NUM_OPTIONS)])
        for i in range(NUM_OPTIONS):
            prompt = prompt.replace(f'<txt{i}>', txt[i])

        # Get the prompt ending from default args
        prompt_end = self.prompt_template.split(".")[-1]
        prompt += "".join(prompt_end.split(",")[:-1])

        if self.select_by == "generate": 
            prompt += " Which option is correct, "
            prompt += " or ".join(options)
            gt = options[gt]
        elif self.select_by == "loss_score":
            gt = txt[gt]
        else:
            raise NotImplementedError("Selection method not implemented.")

        prompt += "?"
        
        return self.get_image_by_id(self.images[idx]), prompt, gt, txt 

    def get_image_by_id(self, image_id): 
        vg_root = '/dccstor/leonidka1/victor_space/data/crepe'
        vg_image_paths = [f'{vg_root}/VG_100K', f'{vg_root}/VG_100K_2']

        # Check either dir
        for p in vg_image_paths:
            path = os.path.join(p, f"{image_id}.jpg")
            if os.path.exists(path):
                return path 

        return None

    def __len__(self):
        return len(self.captions)


def get_benchmark_dataset(
    args,
    prompt_template=None,
    dataset=None,
):

    data_dict = {}
    if args is not None:
        dataset = args.eval_dataset

    # seed
    max_int_32bit = 2 ** 32 - 1
    SEED = int(round(time.time() * 1000)) % max_int_32bit
    print(f'Setting seed to: {SEED}')

    if dataset == "VL_Checklist":
        options = ['A', 'B']
        for k in tqdm(VLC, desc="Loading VL Checklist Dataset"):
            data_k = []
            for p in glob.glob(VLC[k]):
                ds_name = os.path.basename(os.path.dirname(p))
                with open(p, 'r') as f:
                    dd = json.load(f)
                    new_dd = []
                    for d in dd:
                        d[0] = os.path.join(VLC_ROOT, ds_name, d[0])
                        # Prepare the prompt here
                        opt_txt = np.array([d[1]['POS'][0], d[1]['NEG'][0]])
                        ixx = np.arange(len(opt_txt))
                        np.random.shuffle(ixx)
                        gt = np.where(ixx == 0)[0][0]
                        txt = opt_txt[ixx]

                        # Construct the prompt
                        if prompt_template is None:
                            prompt = txt[0] + ", " + txt[1]
                        else:
                            prompt = prompt_template.replace('<txt1>', txt[0]).replace('<txt2>', txt[1])

                        if args.select_by == "generate": 
                            prompt += ", A or B?"  
                            gt = options[gt]
                        elif args.select_by == "loss_score":
                            prompt += "?"
                            gt = txt[gt]
                        else:
                            raise NotImplementedError("Selection method not implemented.")

                        new_dd.append([d[0], prompt, gt, txt])
                         
                    data_k.extend(new_dd)
            data_dict[k] = data_k

    elif dataset == "CREPE":
        options = [chr(i) for i in range(65, 65 + args.num_neg_options + 1)]
        hard_neg_opts = ['atom', 'negate', 'swap']
        file_indices = list(range(4, 13))
        eval_options = list(itertools.product(hard_neg_opts, file_indices))

        for eo in tqdm(eval_options, desc="Loading CREPE Dataset"):
            hard_neg_type = eo[0]
            idx = eo[1]
            retrieval_data_path = os.path.join(f'/dccstor/leonidka1/victor_space/benchmarks/crepe/data/prod_hard_negatives/{hard_neg_type}/prod_vg_hard_negs_{hard_neg_type}_complexity_{idx}.csv')
            
            data_dict[hard_neg_type] = BaseCsvDataset(args, retrieval_data_path, prompt_template)

    elif dataset == "SUGAR":
        options = ['A', 'B']
        coco_data_root = "/dccstor/leonidka1/victor_space/data/coco/images/val2017"
        sugar_data_root = "/dccstor/leonidka1/victor_space/benchmarks/sugar-crepe/data"
        # coco_data_root = "/data1/lin/data/LMM_benchmarks/SugarCrepe/val2017"    #  to change
        # sugar_data_root = "/data1/lin/data/LMM_benchmarks/SugarCrepe"

        # coco_data_root = "/system/user/publicdata/LMM_benchmarks/SugarCrepe/val2017"    #  to change
        # sugar_data_root = "/system/user/publicdata/LMM_benchmarks/SugarCrepe"

        test_options = {
            # 'add_obj'    : f'{sugar_data_root}/add_obj.json',
            # 'add_att'    : f'{sugar_data_root}/add_att.json',
            'replace_obj': f'{sugar_data_root}/replace_obj.json',
            'replace_att': f'{sugar_data_root}/replace_att.json',
            'replace_rel': f'{sugar_data_root}/replace_rel.json',
            # 'swap_obj'   : f'{sugar_data_root}/swap_obj.json',
            # 'swap_att'   : f'{sugar_data_root}/swap_att.json',
        }
        seen_img = set()
        for c, data_path in tqdm(test_options.items(), desc="Loading Sugar-CREPE Dataset"):
            loaded_json = json.load(open(data_path, 'r', encoding='utf-8'))
            new_items = []

            for key, value in loaded_json.items():
                if args.num_samples and len(new_items) >= args.num_samples: # for experimenting with only 50 samples first
                    break

                image_path = f"{coco_data_root}/{value['filename']}"
                if image_path in seen_img: # avoid duplicate images
                    continue
                else:
                    seen_img.add(image_path)

                if os.path.exists(image_path):
                    opt_txt = np.array([value['caption'].strip(), value['negative_caption'].strip()])
                    ixx = np.arange(len(opt_txt))
                    np.random.shuffle(ixx)
                    gt = np.where(ixx == 0)[0][0]
                    txt = opt_txt[ixx]

                    prompt = "What is a suitable caption for this image?"
                    if args.select_by == "generate":
                        gt = options[gt]
                    elif args.select_by == "loss_score" or args.select_by == 't2i':
                        gt = txt[gt]
                    else:
                        raise NotImplementedError("Selection method not implemented.")

                    new_items.append([image_path, prompt, gt, txt, key])
            data_dict[c] = new_items
            print(len(new_items))

    elif dataset == "LN_SUBSET":
        options = ['A', 'B']
        ln_data_root = "/dccstor/leonidka1/irenespace/data/localized_narratives_subset_img"
        new_items = []

        for img_path in sorted(os.listdir(ln_data_root)):
            new_items.append([f'{ln_data_root}/{img_path}', None, None, None, "ln_subset"])

        data_dict["ln_subset"] = new_items
        print(len(new_items))

    elif dataset == "LN_LESS_NOUNS":
        options = ['A', 'B']
        ln_data_root = "/dccstor/leonidka1/irenespace/data/localized_narratives_less_nouns"
        new_items = []

        for img_dir in sorted(os.listdir(ln_data_root)):
            try:
                if int(img_dir) not in set(range(5,11)):
                    continue

                num_imgs = 0
                for img_path in sorted(os.listdir(os.path.join(ln_data_root, img_dir))):
                    if num_imgs == 10:
                        break
                    new_items.append([f'{ln_data_root}/{img_dir}/{img_path}', None, None, None, "ln_less_nouns"])
                    num_imgs += 1
            except:
                continue

        data_dict["ln_less_nouns"] = new_items
        print(len(new_items))

    elif dataset == "LN_FINETUNE":
        options = ['A', 'B']
        ln_data_root = "/dccstor/leonidka1/irenespace/data/localized_narratives_finetune"
        new_items = []

        for img_dir in sorted(os.listdir(ln_data_root)):
            try:
                if int(img_dir) not in set(range(5,11)):
                    continue

                for img_path in sorted(os.listdir(os.path.join(ln_data_root, img_dir))):
                    new_items.append([f'{ln_data_root}/{img_dir}/{img_path}', None, None, None, "ln_finetune"])

            except:
                continue

        data_dict["ln_finetune"] = new_items
        print(len(new_items))

    elif dataset == "LN_MOST_NOUNS":
        options = ['A', 'B']
        ln_data_root = "/dccstor/leonidka1/irenespace/data/localized_narratives_20k_images_with_most_nouns"
        new_items = []

        for img_dir in sorted(os.listdir(ln_data_root), reverse=True):
            for img_path in sorted(os.listdir(os.path.join(ln_data_root, img_dir))):
                if len(new_items) >= args.num_samples:
                    break
                new_items.append([f'{ln_data_root}/{img_dir}/{img_path}', None, None, None, "ln_most_nouns"])

        data_dict["ln_most_nouns"] = new_items
        print(len(new_items))

    elif dataset == "GPT4V_VLM_GEN":
        if args.gpt4v_type == 'dim':
            partitions = {
                'vga': '',
                'vgr': '',
                'replace_obj': '',
                'replace_att': f"/dccstor/leonidka1/irenespace/gpt4v_results/negs/dims/debug_step1_{args.gpt4v_dim}/gpt4v_all-samples.csv",
                'replace_rel': "",
            }
        elif args.gpt4v_type == 'desc':
            path = f"/dccstor/leonidka1/irenespace/gpt4v_results/desc/step{args.gpt4v_dataset_step}/{args.experiment_name}/gpt4v_all-samples.csv"
            partitions = {
                'replace_obj': path,
                'replace_att': path,
                'replace_rel': path,
                'ln_subset': path,
                'ln_less_nouns': path,
                'ln_finetune': path
            }
        else:
            print(f'Generation type {args.gen_type} not yet implemented.')

        for c, data_path in tqdm(partitions.items(), desc="Loading GPT4V_VLM_GEN Dataset"):

            if args.dataset_partition is not None and c != args.dataset_partition:
                continue

            df_partition = pd.read_csv(data_path)
            options = None # A or B for other datasets in multiple choice format
            new_items = []
            seen = set()

            for ind, value in df_partition.iterrows():
                image_path = f"{value['image_path']}"
                if os.path.exists(image_path):
                    if value['orig_ind'] in seen:
                        continue
                    else:
                        seen.add(value['orig_ind'])

                    if args.select_by == "generate":
                        # each item in new_items has 0 the original sample index, 1 image path, 2 orig pos, and 3 orig neg,
                        # 4 original gpt4v response, 5 q1 question, 6 q2 question, 7 q3 question
                        curr = [
                            value['orig_ind'], image_path, value['orig_pos'], value['orig_neg'],
                            value[f'step{args.gpt4v_dataset_step}_gpt4v_response']
                        ]
                        if args.gpt4v_type == 'dim':
                            curr += [value['step1_q1_q'], value['step1_q2_q'], value['step1_q3_q']]
                        elif args.gpt4v_type == 'desc':
                            if args.gpt4v_dataset_step == 1:
                                curr += ["Please describe this image in as much detail as possible. Include everything you see, and be as " +
                                     "specific as possible, such as describing objects, attributes, locations, lighting..."]
                            elif args.gpt4v_dataset_step == 3:
                                for q_ind in range(1,11):
                                    # just in case, quality control--exclude any incorrectly formatted samples
                                    if str(value[f'step{args.gpt4v_dataset_step}_q{q_ind}_q']) in {'nan', ''}:
                                        continue
                                    curr += [value[f'step{args.gpt4v_dataset_step}_q{q_ind}_q']]
                        new_items.append(curr)

                    elif args.select_by == "loss_score" or args.select_by == "t2i":
                        # each item in new_items has 0 the original sample index, 1 image path, 2 orig pos, and 3 orig neg,
                        # 4 prompt, 5 ground truth (correct answer), 6 new negative, 7 generated question index, 8 generated negative index
                        # 9 the two text options for A and B choices

                        for q_ind in range(1, 11):
                            duplicate_negs = set()
                            curr_q, curr_pos = value[f'step1_q{q_ind}_q'], value[f'step1_q{q_ind}_a']

                            for n_ind in range(1, 6):

                                # skip duplicate negatives or empty/blank strings
                                curr_neg = value[f'step1_q{q_ind}_n{n_ind}']
                                if curr_neg in duplicate_negs or str(curr_neg) in {'nan', ''}:
                                    continue
                                else:
                                    duplicate_negs.add(curr_neg)

                                # shuffle the order of the text options
                                opt_txt = np.array([curr_pos, curr_neg])
                                ixx = np.arange(len(opt_txt))
                                np.random.shuffle(ixx)
                                gt = np.where(ixx == 0)[0][0]
                                txt = opt_txt[ixx]

                                new_items.append([
                                    value['orig_ind'], image_path, value['orig_pos'], value['orig_neg'],
                                    curr_q, txt[gt], curr_neg, q_ind, n_ind, txt])
            data_dict[c] = new_items

    elif dataset == "GPT4V_EVAL":
        if args.gpt4v_type == 'dim':
            partitions = {
                'vga': '',
                'vgr': '',
                'replace_obj': '',
                'replace_att': f"/dccstor/leonidka1/irenespace/gpt4v_results/negs/dims/debug_step{args.gpt4v_dataset_step}_{args.gpt4v_dim}/gpt4v_all-samples.csv",
                'replace_rel': "",
            }
        elif args.gpt4v_type == 'desc':
            path = f"/dccstor/leonidka1/irenespace/gpt4v_results/desc/step{args.gpt4v_dataset_step}/{args.experiment_name}/gpt4v_all-samples.csv"
            partitions = {
                'replace_obj': path,
                'replace_att': path,
                'replace_rel': path,
                'ln_subset': path,
                'ln_less_nouns': path,
                'ln_finetune': path
            }
        else:
            print(f'Generation type {args.gen_type} not yet implemented.')

        for c, data_path in tqdm(partitions.items(), desc=f"Loading GPT4V_EVAL Dataset (produced from step {args.gpt4v_dataset_step})"):

            if args.dataset_partition is not None and c != args.dataset_partition:
                continue

            df_partition = pd.read_csv(data_path)
            options = ['A', 'B']
            new_items = []
            seen = set()

            for ind, value in df_partition.iterrows():
                image_path = f"{value['image_path']}"
                if os.path.exists(image_path):
                    if value['orig_ind'] in seen:
                        continue
                    else:
                        seen.add(value['orig_ind'])

                    for q_ind in range(1, 11):
                        duplicate_negs = set()
                        curr_q, curr_pos = value[f'step{args.gpt4v_dataset_step}_q{q_ind}_q'], value[
                            f'step{args.gpt4v_dataset_step}_q{q_ind}_a']

                        # just in case, quality control--exclude any incorrectly formatted samples
                        if str(curr_pos) in {'nan', ''}:
                            continue

                        for n_ind in range(1, 6):

                            # skip duplicate negatives or empty/blank strings
                            curr_neg = value[f'step{args.gpt4v_dataset_step}_q{q_ind}_n{n_ind}']
                            if curr_neg in duplicate_negs or str(curr_neg) in {'nan', ''}:
                                continue
                            else:
                                duplicate_negs.add(curr_neg)

                            # shuffle the order of the text options
                            opt_txt = np.array([curr_pos, curr_neg])
                            ixx = np.arange(len(opt_txt))
                            np.random.shuffle(ixx)
                            gt = np.where(ixx == 0)[0][0]
                            txt = opt_txt[ixx]

                            if args.select_by == "generate":
                                gt = options[gt]
                            elif args.select_by == "loss_score" or args.select_by == 't2i':
                                gt = txt[gt]

                            new_items.append([
                                value['orig_ind'], image_path, value['orig_pos'], value['orig_neg'],
                                value[f'step{args.gpt4v_dataset_step}_gpt4v_response'], curr_q, gt, curr_neg,
                                q_ind, n_ind, txt])

            data_dict[c] = new_items

    elif dataset == 'GPT4V_FILTER':
        partitions = {
            p: f"/dccstor/leonidka1/irenespace/gpt4v_results/desc/step7/{p}{'_333' if 'replace' in p else ''}/llava-v1.6-7b/{args.select_by}/filter_results_dataframe_count-3.csv"
            for p in {'replace_att', 'replace_obj', 'replace_rel', 'ln_subset', 'ln_less_nouns', 'ln_finetune'}
        }
        options = ['A', 'B']
        for c, data_path in tqdm(partitions.items(), desc=f"Loading GPT4V_FILTER Dataset (produced from step 7 and post filtering)"):

            if args.dataset_partition is not None and c != args.dataset_partition:
                continue

            df_partition = pd.read_csv(data_path).sort_values(by=['orig_id', 'q_ind', 'n_ind'])
            df_questions = pd.read_csv(f"/dccstor/leonidka1/irenespace/gpt4v_results/desc/step6/{c}{'_333' if 'replace' in c else ''}/gpt4v_all-samples.csv").sort_values(by=['orig_ind'])

            options = ['A', 'B'] # A or B for other datasets in multiple choice format
            new_items = []

            for ind, value in df_partition.iterrows():
                image_path = f"{value['image']}"
                if os.path.exists(image_path):

                    # each item in new_items has 0 the original sample index, 1 image path, 2 orig pos, and 3 orig neg,
                    # 4 original gpt4v response, 5 a question, 6 a correct answer, 7 new negative, 8 generated question index,
                    # 9 generated negative index, 10 the two text options for A and B choices
                    id, question, correct, neg = value['orig_id'], value['question'], value['correct'], value['neg']
                    q_ind, n_ind = value['q_ind'], value['n_ind']
                    question_data = df_questions[(df_questions['orig_ind'] == id) & (df_questions['image_path'] == image_path)]
                    pos = question_data[f'step6_q{q_ind}_a'].item()
                    assert question_data[f'step6_q{q_ind}_n{n_ind}'].item() == neg

                    # shuffle the order of the text options
                    opt_txt = np.array([pos, neg])
                    ixx = np.arange(len(opt_txt))
                    np.random.shuffle(ixx)
                    gt = np.where(ixx == 0)[0][0]
                    txt = opt_txt[ixx]

                    if args.select_by == "generate":
                        gt = options[gt]
                    elif args.select_by == "loss_score" or args.select_by == 't2i':
                        gt = txt[gt]

                    new_items.append([
                        id, image_path, value['orig_pos'], value['orig_neg'],
                        value[f'step3_gpt4v_response'], question, gt, neg, q_ind, n_ind, txt
                    ])

            data_dict[c] = new_items

    elif dataset == "SUGAR_PROMPT":
        options = ['A', 'B']
        coco_data_root = "/dccstor/leonidka1/victor_space/data/coco/images/val2017"
        sugar_data_root = "/dccstor/leonidka1/victor_space/benchmarks/sugar-crepe/data"

        test_options = {
            # 'add_obj'    : f'{sugar_data_root}/add_obj.json',
            # 'add_att'    : f'{sugar_data_root}/add_att.json',
            # 'replace_obj': f'{sugar_data_root}/replace_obj.json',
            # 'replace_att': f'{sugar_data_root}/replace_att.json',
            'replace_rel': "/dccstor/leonidka1/irenespace/scratch/prompt_positives/sugar_rephrase/20231025_034036/falcon-180b_replace_rel_100-samples.csv",
            # 'swap_obj'   : f'{sugar_data_root}/swap_obj.json',
            # 'swap_att'   : f'{sugar_data_root}/swap_att.json',
        }
        for c, data_path in tqdm(test_options.items(), desc="Loading Sugar-CREPE Dataset"):

            sugar_correct_df = pd.read_csv(data_path)
            new_items = []
            for ind, value in sugar_correct_df.iterrows():

                image_path = f"{value['image_path']}"
                if os.path.exists(image_path):
                    opt_txt = np.array([value['pos'], value['new_neg']])
                    ixx = np.arange(len(opt_txt))
                    np.random.shuffle(ixx)
                    gt = np.where(ixx == 0)[0][0]
                    txt = opt_txt[ixx]

                    # Construct the prompt
                    if prompt_template is None:
                        prompt = txt[0] + ", " + txt[1]
                    else:
                        prompt = prompt_template.replace('<txt1>', txt[0]).replace('<txt2>', txt[1])

                    if args.select_by == "generate":
                        prompt += ", A or B?"
                        gt = options[gt]
                    elif args.select_by == "loss_score":
                        # prompt += "?"
                        prompt = "What is a suitable caption for this image?"
                        gt = txt[gt]
                    else:
                        raise NotImplementedError("Selection method not implemented.")

                    new_items.append([image_path, prompt, gt, txt])
            data_dict[c] = new_items

    elif dataset == "SAMPLE_NEG":
        options = ['A', 'B']

        test_options = {
            'vga': '/dccstor/leonidka1/irenespace/scratch/prompt_negatives/aro/20231112_0035/falcon-180b_vga_all-samples.csv',
            'vgr': '/dccstor/leonidka1/irenespace/scratch/prompt_negatives/aro/20231112_1356/falcon-180b_vgr_all-samples.csv',
            'replace_obj': '/dccstor/leonidka1/irenespace/scratch/prompt_negatives/sugar/20231105_2219/falcon-180b_replace_obj_all-samples.csv',
            'replace_att': "/dccstor/leonidka1/irenespace/scratch/prompt_negatives/sugar/20231105_1456/falcon-180b_replace_att_all-samples.csv",
            'replace_rel': "/dccstor/leonidka1/irenespace/scratch/prompt_negatives/sugar/20231111_1453/falcon-180b_replace_rel_all-samples.csv",
        }
        for c, data_path in tqdm(test_options.items(), desc="Loading SAMPLE_NEG (generated hard negatives) Dataset"):

            if args.dataset_partition is not None and c != args.dataset_partition:
                continue

            sugar_correct_df = pd.read_csv(data_path).sort_values(by=['orig_ind'])

            # each item in new_items has 0 the original sample index, 1 image path, 2 orig pos, and 3 orig neg,
            # followed by 3 elements for each generated response: a prompt, ground truth (original) positive,
            # and the two text options for A and B choices (4 - 18)
            new_items = []
            seen = set()

            for ind, value in sugar_correct_df.iterrows():
                image_path = f"{value['image_path']}"
                if os.path.exists(image_path):
                    if value['orig_ind'] in seen:
                        continue
                    else:
                        seen.add(value['orig_ind'])

                    new_items.append([value['orig_ind'], image_path, value['orig_pos'], value['orig_neg']])

                    for gen_ind in range(5):
                        opt_txt = np.array([value['orig_pos'], value[f'new_neg_{gen_ind}']])
                        ixx = np.arange(len(opt_txt))
                        np.random.shuffle(ixx)
                        gt = np.where(ixx == 0)[0][0]
                        txt = opt_txt[ixx]

                        # Construct the prompt
                        if prompt_template is None:
                            prompt = txt[0] + ", " + txt[1]
                        else:
                            prompt = prompt_template.replace('<txt1>', txt[0]).replace('<txt2>', txt[1])

                        if args.select_by == "generate":
                            prompt += ", A or B?"
                            gt = options[gt]
                        elif args.select_by == "loss_score":
                            # prompt += "?"
                            prompt = "What is a suitable caption for this image?"
                            gt = txt[gt]
                        else:
                            raise NotImplementedError("Selection method not implemented.")

                        new_items[-1] += [prompt, gt, txt]
            data_dict[c] = new_items

    elif dataset == "SAMPLE_NEG_UPDATED":
        options = ['A', 'B']

        if args.prompt_model == 'llm':
            test_options = {
                'vga': '',
                'vgr': '',
                'replace_obj': '/dccstor/leonidka1/irenespace/llm_results_updated/negs/SUGAR_replace_obj/mixtral-8x7b-instruct-v01-q_all-samples.csv',
                'replace_att': "/dccstor/leonidka1/irenespace/llm_results_updated/negs/SUGAR_replace_att/mixtral-8x7b-instruct-v01-q_all-samples.csv",
                'replace_rel': "/dccstor/leonidka1/irenespace/llm_results_updated/negs/SUGAR_replace_rel/mixtral-8x7b-instruct-v01-q_all-samples.csv",
            }
        elif args.prompt_model == 'vlm':
            # for vlm results
            test_options = {
                'vga': '',
                'vgr': '',
                'replace_obj': '/dccstor/leonidka1/irenespace/vlm_results/negs/SUGAR_replace_obj/llava-v1.6-7b_all-samples.csv',
                'replace_att': "/dccstor/leonidka1/irenespace/vlm_results/negs/SUGAR_replace_att/llava-v1.6-7b_all-samples.csv",
                'replace_rel': "/dccstor/leonidka1/irenespace/vlm_results/negs/SUGAR_replace_rel/llava-v1.6-7b_all-samples.csv",
            }
        else:
            print('Invalid argument for prompt model.')

        for c, data_path in tqdm(test_options.items(), desc="Loading SAMPLE_NEG_UPDATED Dataset (generated hard negatives separately)"):

            if args.dataset_partition is not None and c != args.dataset_partition:
                continue

            sugar_correct_df = pd.read_csv(data_path).sort_values(by=['orig_ind'])

            # each item in new_items has 0 the original sample index, 1 image path, 2 orig pos, and 3 orig neg,
            # 4 prompt, 5 ground truth (original) positive, 6 new negative, 7 new change, 8 contradiction check for the new neg,
            # 9 the two text options for A and B choices
            new_items = []
            seen = set()

            for ind, value in sugar_correct_df.iterrows():
                image_path = f"{value['image_path']}"
                if os.path.exists(image_path):
                    if value['orig_ind'] in seen:
                        continue
                    else:
                        seen.add(value['orig_ind'])

                    duplicate_negs = set()
                    for gen_ind in range(5):
                        # skip duplicate negatives or empty/blank strings
                        if value[f'new_neg_{gen_ind}'] in duplicate_negs or str(value[f'new_neg_{gen_ind}']) in {'nan', ''}:
                            continue
                        else:
                            duplicate_negs.add(value[f'new_neg_{gen_ind}'])

                        opt_txt = np.array([value['orig_pos'], value[f'new_neg_{gen_ind}']])
                        ixx = np.arange(len(opt_txt))
                        np.random.shuffle(ixx)
                        gt = np.where(ixx == 0)[0][0]
                        txt = opt_txt[ixx]

                        prompt = "What is a suitable caption for this image?"
                        # prompt = 'Which option is correct for this image?'
                        if args.select_by == "generate":
                            gt = options[gt]
                        elif args.select_by == "loss_score" or args.select_by == "t2i":
                            gt = txt[gt]
                        else:
                            raise NotImplementedError("Selection method not implemented.")

                        new_items.append([value['orig_ind'], image_path, value['orig_pos'], value['orig_neg'], prompt, gt, value[f'new_neg_{gen_ind}'], value[f'new_change_{gen_ind}'], value[f'new_contradict_{gen_ind}'], txt])

            data_dict[c] = new_items

    elif dataset == "MODEL_SPEC_UPDATED_LLM":
        options = ['A', 'B']
        filter_negs_root = f"/dccstor/leonidka1/irenespace/{args.prompt_model}_results_updated/filter/{args.model}/{args.filter_dataset}/" # change the ending subfolder
        # filter_negs_root = f"/dccstor/leonidka1/irenespace/{args.prompt_model}_results_updated/filter/{args.model}/perp_and_gen/"  # change the ending subfolder

        test_options = {}
        if args.filter_dataset == 'SAMPLE_NEG_UPDATED':
            for part in ['vga', 'vgr', 'replace_obj', 'replace_att', 'replace_rel']:
                test_options[part] = f"{filter_negs_root}/ed_SAMPLE_NEG_UPDATED_ev_{args.model}_dp_{part}/{args.model}_all-samples.csv"
        elif args.filter_dataset == 'llava-mix':
            test_options['llava-mix'] = f"{filter_negs_root}/ed_llava_mix_ev_{args.model}/{args.model}_all-samples.csv"

        for c, data_path in tqdm(test_options.items(), desc="Loading Model Specific Updated Dataset"):
            if args.dataset_partition is not None and c != args.dataset_partition:
                continue

            partition_df = pd.read_csv(data_path).sample(frac=1, random_state=SEED)
            new_items = []

            for ind, value in partition_df.iterrows():
                if args.num_samples is not None and len(new_items) >= args.num_samples:
                    break

                image_path = f"{value['image']}" if 'image' in value.keys() else f"{value['image_path']}"

                if type(value['orig_pos']) is str and (len(value['orig_pos']) == 1) and (value['orig_pos'].isalpha()):
                    continue

                if os.path.exists(image_path):
                    if args.which_neg == 'new_neg':  #  our new negatives
                        opt_txt = np.array([value['orig_pos'], value[f'new_neg']])
                    elif args.which_neg == 'orig_neg':  #  baseline
                        opt_txt = np.array([value['orig_pos'], value[f'orig_neg']])
                    else:
                        raise ValueError("which_neg not recognized.")
                    ixx = np.arange(len(opt_txt))
                    np.random.shuffle(ixx)
                    gt = np.where(ixx == 0)[0][0]
                    txt = opt_txt[ixx]

                    if args.filter_dataset == 'SAMPLE_NEG_UPDATED':
                        prompt = "What is a suitable caption for this image?"
                    elif args.filter_dataset == 'llava-mix':
                        prompt = value['question']

                    if args.select_by == "generate":
                        gt = options[gt]
                    elif args.select_by == "loss_score" or args.select_by == "t2i":
                        gt = txt[gt]
                    else:
                        raise NotImplementedError("Selection method not implemented.")

                    if args.filter_dataset == 'SAMPLE_NEG_UPDATED':
                        new_items.append([image_path, prompt, gt, txt, value['orig_id'], value['orig_pos'], value['orig_neg'], value['new_neg'], value['new_change'], value['contradiction_check']])
                    elif args.filter_dataset == 'llava-mix':
                        new_items.append(
                            [image_path, prompt, gt, txt, value['sample_ind'], value['orig_pos'], value['orig_neg']])

            data_dict[c] = new_items

    elif dataset in ["COMBINED_LLM", "LEAVE_ONE_OUT_LLM"]:
        options = ['A', 'B']

        if dataset == "LEAVE_ONE_OUT_LLM":
            aro_root = sugar_root =  f'/data1/lin/data/LMM_benchmarks/CombinedLLM/leave_out_{args.model}'
        elif dataset == "COMBINED_LLM":
            # aro_root = '/dccstor/leonidka1/irenespace/scratch/prompt_negatives/aro/results'
            # sugar_root = '/dccstor/leonidka1/irenespace/scratch/prompt_negatives/sugar/results'
            aro_root = sugar_root =  '/system/user/publicdata/LMM_benchmarks/CombinedLLM'
        else:
            raise ValueError("Dataset not recognized.")

        test_options = {
            'vga': f'{aro_root}/vga_combined.csv',
            'vgr': f'{aro_root}/vgr_combined.csv',
            'replace_obj': f'{sugar_root}/replace_obj_combined.csv',
            'replace_att': f'{sugar_root}/replace_att_combined.csv',
            'replace_rel': f'{sugar_root}/replace_rel_combined.csv',
        }
        for c, data_path in tqdm(test_options.items(), desc="Loading Combined-LLM Dataset"):

            if args.dataset_partition is not None and c != args.dataset_partition:
                continue

            partition_df = pd.read_csv(data_path).sample(frac=1,random_state=SEED)

            # each item in new_items has first the original sample index, image path, orig pos, and orig neg,
            # followed by prompt, ground truth (original) positive, and the two text options for A and B choices
            # and ending with the
            new_items = []

            for ind, value in partition_df.iterrows():
                image_path = f"{value['image']}"
                if args.username == 'wei_lin':
                    if '/data1/lin/data/LMM_benchmarks' in image_path:
                        image_path = image_path.replace('/data1/lin/data/LMM_benchmarks', '/system/user/publicdata/LMM_benchmarks')
                    elif '/data2/lin/data/ARO_benchmark' in image_path:
                        image_path = image_path.replace('/data2/lin/data/ARO_benchmark', '/system/user/publicdata/LMM_benchmarks/ARO_benchmark')

                if os.path.exists(image_path):
                    if args.which_neg == 'new_neg':  #  our new negatives
                        opt_txt = np.array([value['orig_pos'], value[f'new_neg']])
                    elif args.which_neg == 'orig_neg':  #  baseline
                        opt_txt = np.array([value['orig_pos'], value[f'orig_neg']])
                    else:
                        raise ValueError("which_neg not recognized.")
                    ixx = np.arange(len(opt_txt))
                    np.random.shuffle(ixx)
                    gt = np.where(ixx == 0)[0][0]
                    txt = opt_txt[ixx]

                    # Construct the prompt
                    if prompt_template is None:
                        prompt = txt[0] + ", " + txt[1]
                    else:
                        prompt = prompt_template.replace('<txt1>', txt[0]).replace('<txt2>', txt[1])

                    if args.select_by == "generate":
                        prompt += ", A or B?"
                        gt = options[gt]
                    elif args.select_by == "loss_score":
                        prompt = "What is a suitable caption for this image?"
                        gt = txt[gt]
                    else:
                        raise NotImplementedError("Selection method not implemented.")

                    new_items.append([image_path, prompt, gt, txt, value['orig_ind'], value['orig_pos'], value['orig_neg'], value['new_neg'], value['new_change'], value['contradiction_check']])

            data_dict[c] = new_items

    elif dataset == "LLM":
        options = ['A', 'B']

        test_options = {
            'instructblip_flan_t5': {
                'vga': [
                    "/dccstor/leonidka1/irenespace/scratch/prompt_negatives/aro/results/vga_instructblip_flan_t5/score_diffs.csv"
                ],
                'vgr': [
                    "/dccstor/leonidka1/irenespace/scratch/prompt_negatives/aro/results/vgr_instructblip_flan_t5/score_diffs.csv"
                ],
                'replace_obj': [
                    "/dccstor/leonidka1/irenespace/scratch/prompt_negatives/sugar/results/replace_obj_instructblip_flan_t5/score_diffs.csv"
                ],
                'replace_att': [
                    "/dccstor/leonidka1/irenespace/scratch/prompt_negatives/sugar/results/replace_att_instructblip_flan_t5/score_diffs.csv"
                ],
                'replace_rel': [
                    "/dccstor/leonidka1/irenespace/scratch/prompt_negatives/sugar/results/replace_rel_instructblip_flan_t5/score_diffs.csv"
                ],
            },
            'instructblip_vicuna_7b': {
                'vga': [
                    "/dccstor/leonidka1/irenespace/scratch/prompt_negatives/aro/results/vga_instructblip_vicuna_7b/score_diffs.csv"
                ],
                'vgr': [
                    "/dccstor/leonidka1/irenespace/scratch/prompt_negatives/aro/results/vgr_instructblip_vicuna_7b/score_diffs.csv"
                ],
                'replace_obj': [
                    "/dccstor/leonidka1/irenespace/scratch/prompt_negatives/sugar/results/replace_obj_instructblip_vicuna_7b/score_diffs.csv"
                ],
                'replace_att': [
                    "/dccstor/leonidka1/irenespace/scratch/prompt_negatives/sugar/results/replace_att_instructblip_vicuna_7b/score_diffs.csv"
                ],
                'replace_rel': [
                    "/dccstor/leonidka1/irenespace/scratch/prompt_negatives/sugar/results/replace_rel_instructblip_vicuna_7b/score_diffs.csv"
                ],
            },
            'llava-v1.5-7b': {
                'vga': [
                    "/dccstor/leonidka1/irenespace/scratch/prompt_negatives/aro/results/vga_llava-v1.5-7b/score_diffs.csv"
                ],
                'vgr': [
                    "/dccstor/leonidka1/irenespace/scratch/prompt_negatives/aro/results/vgr_llava-v1.5-7b/score_diffs.csv"
                ],
                'replace_obj': [
                    "/dccstor/leonidka1/irenespace/scratch/prompt_negatives/sugar/results/replace_obj_llava-v1.5-7b/score_diffs.csv"
                ],
                'replace_att': [
                    "/dccstor/leonidka1/irenespace/scratch/prompt_negatives/sugar/results/replace_att_llava-v1.5-7b/score_diffs.csv"
                ],
                'replace_rel': [
                    "/dccstor/leonidka1/irenespace/scratch/prompt_negatives/sugar/results/replace_rel_llava-v1.5-7b/score_diffs.csv"
                ],
            }

        }
        for c, data_paths in tqdm(test_options[args.model].items(), desc="Loading LLM Selected Negative Dataset"):

            if args.dataset_partition is not None and c != args.dataset_partition:
                continue

            score_dfs = [pd.read_csv(path) for path in data_paths]

            # each item in new_items has first the original sample index, image path, orig pos, and selected neg,
            # followed by 3 elements for each generated response: a prompt, ground truth (original) positive,
            # and the two text options for A and B choices
            new_items = []

            for ind in range(len(score_dfs[0])):
                rows = [df.iloc[ind] for df in score_dfs]

                orig_id = rows[0]['orig_id']
                image_path = rows[0]['image'] # all rows should have the same image path
                pos = rows[0]['orig_pos']

                # map LLM-generated negative to avg diff across chosen models
                neg2diff = dict()
                for neg_ind in range(5):
                    for row in rows:
                        neg2diff[neg_ind] = neg2diff.setdefault(neg_ind, 0) + row[f'diff_scores_{neg_ind}']
                    neg2diff[neg_ind] /= len(rows) # average of the score diffs

                curr_item = [image_path]
                ordered_negatives = sorted(neg2diff.keys(), key=lambda k: neg2diff[k], reverse=True)

                # select index corresponding to next max avg diff (i.e. most positive loss compared to gt)
                for neg_select_ind in range(args.num_neg_options):
                    neg_select = ordered_negatives[neg_select_ind]
                    prompt = rows[0][f'prompt_{neg_select}']
                    neg = rows[0][f'new_neg_{neg_select}']

                    opt_txt = np.array([pos, neg])
                    ixx = np.arange(len(opt_txt))
                    np.random.shuffle(ixx)
                    gt = np.where(ixx == 0)[0][0]
                    txt = opt_txt[ixx]

                    if args.select_by == "generate":
                        gt = options[gt]
                    elif args.select_by == "loss_score":
                        gt = txt[gt]
                    else:
                        raise NotImplementedError("Selection method not implemented.")

                    curr_item += [prompt, gt, txt]

                curr_item.append(orig_id)
                new_items.append(curr_item)

            data_dict[c] = new_items

    elif dataset == 'llava_mix':
        with open('/dccstor/leonidka1/dev/LLaVA/data/llava_v1_5_mix665k.json', "r") as json_content:
            llava_json = json.load(json_content)

        llava_negs_root = '/dccstor/leonidka1/dev/LLaVA/data/negs/llava_v1_5_mix665k'

        new_items = []
        options = ['A', 'B']
        skip_responses = {'no', 'yes', 'correct', 'wrong', 'true', 'false'}
        item_ind = 0
        for item_dict in tqdm(llava_json, desc="Loading llava_mix Dataset"):

            # if wanting to truncate to a smaller subset of the dataset
            if args.num_samples and item_ind >= args.num_samples:
                # print(f'loaded {args.num_samples} items')
                break

            id = item_dict['id']

            if not os.path.exists(f'{llava_negs_root}/{id}.json'):
                continue
            with open(f'{llava_negs_root}/{id}.json', "r") as file_content:
                try:
                    file = json.load(file_content)

                    # some cases without images
                    if 'image' not in file.keys():
                        continue
                except:
                    # some cases with empty files
                    continue

            num_negs = len(file['conversations']) // 2

            # add prefix to img_path
            ds = file['image'].split('/')[0]
            dirs = {
                'coco': '/dccstor/leonidka1/victor_space/data/coco/images/',
                'gqa': '/dccstor/leonidka1/irenespace/data/gqa/',
                'ocr_vqa': '/dccstor/leonidka1/irenespace/data/ocr_vqa/',
                'textvqa': '/dccstor/leonidka1/irenespace/data/textvqa/',
                'vg': '/dccstor/sivandov1/data/vl_datasets/vg/'
            }
            img_path = dirs[ds] + '/'.join(file['image'].split('/')[1:])

            for ind in range(num_negs):
                human, gpt = file['conversations'][2*ind], file['conversations'][2*ind + 1]
                question = human['value'].removeprefix('<image>').removesuffix('<image>').strip('\n').split('\n')[0]
                POS = gpt['value']
                pos_split = POS.split(' ')

                if str(POS).lower() in skip_responses:
                    continue

                # if bounding box in question or response, skip
                coord_regex = '\[[0-9]+(?:.[0-9]+)?,\s[0-9]+(?:.[0-9]+)?,\s[0-9]+(?:.[0-9]+)?,\s[0-9]+(?:.[0-9]+)?\]'
                if (re.search(coord_regex, question) is not None) or (re.search(coord_regex, POS) is not None):
                    # print statements for debugging
                    # print('Bounding box in question or gt --> skip this sample')
                    # print(question)
                    # print(POS)
                    continue

                # in case the data doesn't actually have any generated negatives, continue on
                if not 'negs' in gpt.keys():
                    continue

                for neg_option in gpt['negs']:

                    # skip if the generative neg is somehow identical to original positive
                    # or if it is a boolean type response
                    if neg_option == POS or str(neg_option).lower() in skip_responses:
                        continue

                    # fill in the rest of the negative with the original sentence structure
                    neg_split = neg_option.split(' ')
                    NEG = ' '.join(neg_split + (pos_split[len(neg_split):] if len(neg_split) < len(pos_split) else []))

                    # Prepare the prompt here
                    opt_txt = np.array([POS, NEG])
                    ixx = np.arange(len(opt_txt))
                    np.random.shuffle(ixx)
                    gt = np.where(ixx == 0)[0][0]
                    txt = opt_txt[ixx]

                    # Construct the prompt
                    prompt = question
                    if args.select_by == "generate":
                        # prompt = f"Option A: <txt1>. Option B: <txt2>. {question.strip('?.,!;:')}, A or B?".replace('<txt1>', txt[0]).replace('<txt2>', txt[1])
                        gt = options[gt]
                    elif args.select_by == "loss_score" or args.select_by == 't2i':
                        # prompt = question
                        gt = txt[gt]
                    else:
                        raise NotImplementedError("Selection method not implemented.")

                    new_items.append([img_path, prompt, gt, txt, item_ind])
                    item_ind += 1

        data_dict["llava_mix"] = new_items

    elif dataset == 'filter_llava_mix':
        # map eval_vlm to csv where the selected challenging negs are
        llava_res_root = '/dccstor/leonidka1/irenespace/prompt_negatives_revised/llava_mix'
        model2negs = {
            'instructblip_flan_t5': f'{llava_res_root}/20240111_2339/instructblip_flan_t5_all-samples.csv',
            'instructblip_vicuna_7b': f'{llava_res_root}//instructblip_vicuna_7b_all-samples.csv',
            'llava_v1.5_7b': f'{llava_res_root}//5_7b_all-samples.csv',
        }
        df = pd.read_csv(model2negs[args.filter_vlm])

        new_items = []
        options = ['A', 'B']
        skip_responses = {'no', 'yes', 'correct', 'wrong', 'true', 'false'}

        for ind, row in tqdm(df.iterrows(), total=len(df), desc="Loading filter_llava_mix Dataset"):

            # if wanting to truncate to a smaller subset of the dataset
            if args.num_samples and ind >= args.num_samples:
                # print(f'loaded {args.num_samples} items')
                break

            item_ind = row['item_ind']
            csv_ind = row['sample_ind']
            img_path = row['image_path']  # all rows should have the same image path
            pos = row['orig_pos']
            neg = row['orig_neg']
            question = row['question']

            if str(pos).lower() in skip_responses or str(neg).lower() in skip_responses:
                continue

            reqs = ast.literal_eval(row['select_reqs'])

            gen_prompt = reqs[f'{args.filter_vlm}_generate']['prompt']
            if 'llava' in args.model:
                gen_prompt += ' Answer with the option’s letter from the given choices directly.'
            ppl_prompt = reqs[f'{args.filter_vlm}_loss_score']['prompt']

            opt_txt = np.array([pos, neg])
            ixx = np.arange(len(opt_txt))
            np.random.shuffle(ixx)
            gt = np.where(ixx == 0)[0][0]
            txt = opt_txt[ixx]

            if args.select_by == "generate":
                gt = options[gt]
            elif args.select_by == "loss_score":
                gt = txt[gt]
            else:
                raise NotImplementedError("Selection method not implemented.")

            new_items.append([img_path, question, gt, txt, neg, item_ind, csv_ind, gen_prompt, ppl_prompt])
            item_ind += 1

        data_dict["filter_llava_mix"] = new_items

    elif dataset == "SEED": # seed-bench wip
        print('dataset: ', dataset) # for debugging
        # seed_data_root = "/dccstor/leonidka1/irenespace/data/seed"
        # seed_cc3m_root = f"{seed_data_root}/seed_bench_image"
        seed_data_root = "/data1/lin/data/LMM_benchmarks/SEED-Bench/"
        seed_cc3m_root = f"{seed_data_root}/SEED-Bench-image"
        seed_json = get_seed_image_data(seed_data_root=seed_data_root, seed_cc3m_root = seed_cc3m_root) # list of questions


        new_items = {i: [] for i in range(1, 10)} # map each image dimension 1-9 to a list of items

        for value in tqdm(seed_json, desc="Loading SEED-Bench Dataset"):
            data_typ = value['data_type']
            dim = value["question_type_id"]

            if data_typ == 'image':

                # DEBUGGING SEED-BENCH: dimension 2 only
                if dim != 7:
                    continue

                image_path = f"{seed_cc3m_root}/SEED-Bench-image/{value['data_id']}"
                if os.path.exists(image_path):
                    prompt = 'Question: ' + value['question'] + '\n\nOptions:\n'

                    pos = value['answer'].lower()
                    neg_options = [value[f"choice_{letter}"] for letter in {'a', 'b', 'c', 'd'} if letter != pos]
                    num_options = len(neg_options) + 1
                    options = [chr(i) for i in range(97, 97 + num_options)] # lower case letter options

                    opt_txt = np.array([value[f"choice_{pos}"]] + neg_options)
                    ixx = np.arange(len(opt_txt))
                    np.random.shuffle(ixx)
                    gt = np.where(ixx == 0)[0][0]
                    txt = opt_txt[ixx]

                    # Construct the prompt
                    prompt += "\n".join([f"({chr(i)}) {txt[i - 97]}" for i in range(97, 97 + num_options)])
                    # prompt += '\n\nShort answer:'
                    prompt += '\n\nAnswer:'

                    if args.select_by == "generate":
                        gt = options[gt]
                    elif args.select_by == "loss_score":
                        gt = txt[gt]
                    else:
                        raise NotImplementedError("Selection method not implemented.")

                    # prompt = value['question'] # FOR DEBUGGING SEED-BENCH
                    prompt = f"""Question: {value['question']}\nAnswer:"""
                    new_items[dim].append([image_path, prompt, gt, txt])

            elif data_typ == 'video':
                continue # skipping video type data for now

            else:
                raise ValueError('Data type is not valid.')

        # data_dict maps each image dimension (1-9) to its list of samples
        for key, val in new_items.items():
            data_dict[f'seed_dim_{key}'] = val

    elif dataset == "ARO_VGR":
        options = ['A', 'B'] 
        visual_genome_root_dir = '/dccstor/leonidka1/victor_space/data/visual_genome'
        # visual_genome_root_dir = '/data2/lin/data/ARO_benchmark/Visual_genome'
        # visual_genome_root_dir = '/system/user/publicdata/LMM_benchmarks/ARO_benchmark/Visual_genome'

        dataset = VG_Relation(root_dir=visual_genome_root_dir)
        
        new_items = []
        ind = 0
        for item_dict in tqdm(dataset.dataset, desc="Loading ARO VGR dataset."):
            POS = item_dict['true_caption']
            NEG = item_dict['false_caption']

            # Prepare the prompt here
            opt_txt = np.array([POS, NEG])
            ixx = np.arange(len(opt_txt))
            np.random.shuffle(ixx)
            gt = np.where(ixx == 0)[0][0]
            txt = opt_txt[ixx]

            # Construct the prompt
            # if prompt_template is None:
            #     prompt = txt[0] + ", " + txt[1]
            # else:
            #     prompt = prompt_template.replace('<txt1>', txt[0]).replace('<txt2>', txt[1])

            # prompt = 'Which option accurately describes the image?'
            # prompt = 'Which option, A or B, best matches the image?'
            prompt = 'Which of the following options more accurately describes this image, A or B?'
            # prompt = 'What is an accurate description for this image, A or B?'
            # prompt = "What is a suitable caption for this image?"
            if args.select_by == "generate": 
                # prompt += ", A or B?"
                gt = options[gt]
            elif args.select_by == "loss_score" or args.select_by == "t2i":
                # prompt = "What is a suitable caption for this image?"
                gt = txt[gt]
            else:
                raise NotImplementedError("Selection method not implemented.")

            new_items.append([item_dict["image_path"], prompt, gt, txt, ind])
            ind += 1

        data_dict["vgr"] = new_items

    elif dataset =="ARO_VGA":
        options = ['A', 'B'] 
        visual_genome_root_dir = '/dccstor/leonidka1/victor_space/data/visual_genome'
        # visual_genome_root_dir = '/data2/lin/data/ARO_benchmark/Visual_genome'
        # visual_genome_root_dir = '/system/user/publicdata/LMM_benchmarks/ARO_benchmark/Visual_genome'
        dataset = VG_Attribution(root_dir=visual_genome_root_dir)

        new_items = []
        ind = 0
        for item_dict in tqdm(dataset.dataset, desc="Loading ARO VGA dataset."):
            POS = item_dict['true_caption']
            NEG = item_dict['false_caption']
            # Prepare the prompt here
            opt_txt = np.array([POS, NEG])
            ixx = np.arange(len(opt_txt))
            np.random.shuffle(ixx)
            gt = np.where(ixx == 0)[0][0]
            txt = opt_txt[ixx]

            # Construct the prompt
            # if prompt_template is None:
            #     prompt = txt[0] + ", " + txt[1]
            # else:
            #     prompt = prompt_template.replace('<txt1>', txt[0]).replace('<txt2>', txt[1])

            # prompt = 'Which option accurately describes the image?'
            # prompt = 'Which option, A or B, best matches the image?'
            # prompt = 'What is an accurate description for this image, A or B?'
            prompt = 'Which of the following options more accurately describes this image, A or B?'
            # prompt = "What is a suitable caption for this image?"
            if args.select_by == "generate": 
                # prompt += ", A or B?"
                gt = options[gt]
            elif args.select_by == "loss_score" or args.select_by == "t2i":
                # prompt = "What is a suitable caption for this image?"
                gt = txt[gt]
            else:
                raise NotImplementedError("Selection method not implemented.")

            new_items.append([item_dict["image_path"], prompt, gt, txt, ind])
            ind += 1

        data_dict["vga"] = new_items

    elif dataset == "COCO":
        options = [chr(i) for i in range(65, 65 + args.num_neg_options + 1)]
        coco_root = '/dccstor/leonidka1/victor_space/data/coco'
        coco_image_root = '/dccstor/leonidka1/victor_space/data/coco/images/'
        dataset = COCO_Order(args=args, root_dir=coco_root)

        new_items = []
        ind = 0
        for item_dict in tqdm(dataset.dataset, desc="Loading COCO dataset."):
            POS = item_dict["true_caption"]
            neg_list = item_dict["false_captions"]
            NEG_LIST = list(np.random.choice(neg_list, size=args.num_neg_options, replace=False))
            NUM_OPTIONS = len(NEG_LIST) + 1

            opt_txt = np.array([POS] + NEG_LIST)
            ixx = np.arange(len(opt_txt))
            np.random.shuffle(ixx)
            gt = np.where(ixx == 0)[0][0]
            txt = opt_txt[ixx]
            prompt = " ".join([f"Option {chr(i)}: <txt{i - 65}>." for i in range(65, 65 + NUM_OPTIONS)])
            for i in range(NUM_OPTIONS):
                prompt = prompt.replace(f'<txt{i}>', txt[i])
            
            # Get the prompt ending from default args
            prompt_end = prompt_template.split(".")[-1]
            prompt_end_without_opts = "".join(prompt_end.split(",")[:-1])
            prompt += prompt_end_without_opts + ", "
            prompt += " or ".join([chr(i) for i in range(65, 65 + NUM_OPTIONS)])
            prompt += "?"

            if args.select_by == "generate": 
                prompt += ", A or B?"  
                gt = options[gt]
            elif args.select_by == "loss_score":
                prompt = "What is a suitable caption for this image?"
                gt = txt[gt]
            else:
                raise NotImplementedError("Selection method not implemented.")

            file_path = f"{coco_image_root}/{item_dict['image_path']}"
            new_items.append([file_path, prompt, gt, txt, ind])
            ind += 1

        data_dict["coco"] = new_items

    elif dataset == "Flickr":
        options = [chr(i) for i in range(65, 65 + args.num_neg_options + 1)]
        flickr_root = '/dccstor/leonidka1/victor_space/data/flickr'
        flickr_image_root = '/dccstor/leonidka1/victor_space/data/flickr'
        dataset = Flickr30k_Order(args=args, root_dir=flickr_root)

        new_items = []
        ind = 0
        for item_dict in tqdm(dataset.dataset, desc="Loading Flickr dataset."):
            POS = item_dict["true_caption"]
            neg_list = item_dict["false_captions"]
            NEG_LIST = list(np.random.choice(neg_list, size=args.num_neg_options, replace=False))
            NUM_OPTIONS = len(NEG_LIST) + 1

            opt_txt = np.array([POS] + NEG_LIST)
            ixx = np.arange(len(opt_txt))
            np.random.shuffle(ixx)
            gt = np.where(ixx == 0)[0][0]
            txt = opt_txt[ixx]
            prompt = " ".join([f"Option {chr(i)}: <txt{i - 65}>." for i in range(65, 65 + NUM_OPTIONS)])
            for i in range(NUM_OPTIONS):
                prompt = prompt.replace(f'<txt{i}>', txt[i])
            
            # Get the prompt ending from default args
            prompt_end = prompt_template.split(".")[-1]
            prompt_end_without_opts = "".join(prompt_end.split(",")[:-1])
            prompt += prompt_end_without_opts + ", "
            prompt += " or ".join([chr(i) for i in range(65, 65 + NUM_OPTIONS)])
            prompt += "?"

            if args.select_by == "generate": 
                prompt += ", A or B?"  
                gt = options[gt]
            elif args.select_by == "loss_score":
                prompt = "What is a suitable caption for this image?"
                gt = txt[gt]
            else:
                raise NotImplementedError("Selection method not implemented.")

            file_path = f"{flickr_image_root}/{item_dict['image_path']}"
            new_items.append([file_path, prompt, gt, txt, ind])
            ind += 1

        data_dict["flickr"] = new_items

    else:
        raise NotImplementedError("This kind of dataset DOESN'T EXIST!")

    return data_dict, options