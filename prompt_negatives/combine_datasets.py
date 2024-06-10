import pandas as pd
import numpy as np

import os
import os.path as osp
from args import get_args

def make_dir(path):
    if not osp.exists(path):
        os.makedirs(path)

vg_dir = '/data2/lin/data/ARO_benchmark/Visual_genome/images'
sugarcrepe_dir = '/data1/lin/data/LMM_benchmarks/SugarCrepe/val2017'
old_vg_dir =  '/dccstor/leonidka1/victor_space/data/visual_genome/images'
old_sugarcrepe_dir = '/dccstor/leonidka1/victor_space/data/coco/images/val2017'
if __name__ == '__main__':
    result_dir = '/data1/lin/data/LMM_benchmarks/CombinedLLM/'

    # MODELS = ['instructblip_flan_t5', 'instructblip_vicuna_7b', 'llava-v1.5-7b']
    # PARTITIONS = {
    #     'sugar': ['replace_rel', 'replace_att', 'replace_obj'],
    #     'aro': ['vga', 'vgr']
    # }
    #
    # for dataset, partitions in PARTITIONS.items():
    #     for p in partitions:
    #         output_file = f'/dccstor/leonidka1/irenespace/scratch/prompt_negatives/{dataset}/results/{p}_combined.csv'
    #
    #         dfs = []
    #         for m in MODELS:
    #             curr = pd.read_csv(f'/dccstor/leonidka1/irenespace/scratch/prompt_negatives/{dataset}/results/{p}_{m}/{p}_{m}_combined_records.csv')
    #             dfs.append(curr[['orig_ind', 'selected_neg_ind', 'orig_pos', 'orig_neg', 'new_neg', 'new_change', 'contradiction_check', 'image']])
    #
    #         df_all = pd.concat(dfs, ignore_index=True)
    #         df_all.to_csv(output_file)


    ######
    # args = get_args()
    #
    # # first, concatenate all generated samples into one updated csv file
    # FOLDER=f'/dccstor/leonidka1/irenespace/llm_results/sim_rouge/{args.dataset_partition}/20240128_1609'
    #
    # dfs = []
    # for csv_path in sorted(os.listdir(FOLDER)):
    #     # if needing to consider subset
    #     # if int(csv_path[-8:-4]) >= 2977:
    #     #     break
    #
    #     if csv_path != f'rouge_{args.dataset_partition}_all-samples.csv':
    #         dfs.append(pd.read_csv(f'{FOLDER}/{csv_path}'))
    #
    # df_all = pd.concat(dfs, ignore_index=True)
    # df_all.to_csv(f'{FOLDER}/rouge_{args.dataset_partition}_all-samples.csv')
    # print(f'results saved to dir {FOLDER}')


    #####
    # Wei's code
    # MODELS = ['instructblip_flan_t5', 'instructblip_vicuna_7b', 'llava-v1.5-7b']
    #
    # # model_to_leave_out =  ['instructblip_flan_t5', 'instructblip_vicuna_7b', 'llava-v1.5-7b'][2]
    # model_only =  ['instructblip_flan_t5', 'instructblip_vicuna_7b', 'llava-v1.5-7b'][2]
    #
    # PARTITIONS = {
    #     'sugar': ['replace_rel', 'replace_att', 'replace_obj'],
    #     'aro': ['vga', 'vgr']
    # }
    #
    # for dataset, partitions in PARTITIONS.items():
    #     # dataset is sugar or aro
    #     for p in partitions:
    #         # output_dir = osp.join(result_dir, f'leave_out_{model_to_leave_out}')
    #         output_dir = osp.join(result_dir, f'only_{model_only}')
    #         make_dir(output_dir)
    #         output_file = osp.join(output_dir, f'{p}_combined.csv')
    #
    #         dfs = []
    #         for m in MODELS:
    #             # if m != model_to_leave_out:
    #             if m == model_only:
    #                 dataset_file = osp.join(result_dir, m, f'{p}_{m}_combined_records.csv' )
    #                 # curr = pd.read_csv(f'/dccstor/leonidka1/irenespace/scratch/prompt_negatives/{dataset}/results/{p}_{m}/{p}_{m}_combined_records.csv')
    #                 curr = pd.read_csv(dataset_file)
    #                 for idx in range(len(curr['image'])):
    #                     imagepath = curr['image'][idx]
    #                     if dataset == 'sugar':
    #                         curr['image'][idx] = curr['image'][idx].replace(old_sugarcrepe_dir, sugarcrepe_dir)
    #                     elif dataset == 'aro':
    #                         curr['image'][idx] = curr['image'][idx].replace(old_vg_dir, vg_dir)
    #                 dfs.append(curr[['orig_ind', 'selected_neg_ind', 'orig_pos', 'orig_neg', 'new_neg', 'new_change', 'contradiction_check', 'image']])
    #####


    # ######
    # ## combining selected negatives for model-specific datasets using revised llm-generation process
    #
    # args = get_args()
    #
    # # first, concatenate all generated samples into one updated csv file
    # FOLDER=f'/dccstor/leonidka1/irenespace/prompt_negatives_revised/aro_vgr/20240102_1750'
    #
    # dfs = []
    # for csv_path in sorted(os.listdir(FOLDER)):
    #     # if needing to consider subset
    #     # if int(csv_path[-8:-4]) >= 2977:
    #     #     break
    #
    #     if csv_path != f'{args.dataset_partition}_{args.model}_all-samples.csv':
    #         curr = pd.read_csv(f'{FOLDER}/{csv_path}').rename(columns={'Unnamed: 0':'sample_ind'})
    #         curr['sample_ind'] = int(csv_path[-8:-4])
    #
    #         if 'select_neg' in curr.columns:
    #             curr = curr.dropna(axis=0, subset='select_neg')
    #             dfs.append(curr[['sample_ind','orig_ind','image_path','orig_pos','orig_neg','select_neg','select_change','select_contradict','select_reqs']])
    #
    # df_all = pd.concat(dfs, ignore_index=True)
    # df_all.to_csv(f'{FOLDER}/{args.dataset_partition}_{args.model}_all-samples.csv')
    # print(f'results saved to dir {FOLDER}')


    ######
    ## combining selected negatives for updated filtering method

    args = get_args()

    # first, concatenate all generated samples into one updated csv file
    # FOLDER = f'/dccstor/leonidka1/irenespace/vlm_results_updated/eval/ed_MODEL_SPEC_UPDATED_LLM_fd_SAMPLE_NEG_UPDATED_ev_llava-v1.6-7b_dp_replace_att_wn_new_neg_s_loss_score_t_0_nt_128_p_11/eval_model_spec/replace_att_llava-v1.6-7b/labels/'
    # FOLDER = f'/dccstor/leonidka1/irenespace/vlm_results_updated/filter/llava-v1.6-7b/SAMPLE_NEG_UPDATED//ed_SAMPLE_NEG_UPDATED_ev_llava-v1.6-7b_dp_replace_obj/'
    # FOLDER = f'/dccstor/leonidka1/irenespace/llm_results_updated/filter/instructblip_flan_t5/ed_SAMPLE_NEG_UPDATED_ev_instructblip_flan_t5_dp_replace_rel/'
    # FOLDER = f'/dccstor/leonidka1/irenespace/llm_results_updated/filter/instructblip_vicuna_7b/ed_SAMPLE_NEG_UPDATED_ev_instructblip_vicuna_7b_dp_replace_rel/'
    FOLDER = f'/dccstor/leonidka1/irenespace/gpt4v_results/desc/step6/ln_finetune/'

    max_num = 0
    dfs = []
    for csv_path in sorted(os.listdir(FOLDER)):
        # if needing to consider subset
        # if int(csv_path[-8:-4]) >= 2977:
        #     break

        if csv_path != f'{args.model}_all-samples.csv':
            max_num = max(max_num, int(csv_path.split('.csv')[0].split('-')[-1]))

            curr = pd.read_csv(f'{FOLDER}/{csv_path}')
            if len(curr) == 0:
                continue

            curr.rename(columns={'Unnamed: 0': 'sample_ind'}, inplace=True)
            curr['sample_ind'] = int(csv_path.split('.csv')[0].split('-')[-1])
            dfs.append(curr)

    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(f'{FOLDER}/{args.model}_all-samples.csv')
    print(f'results saved to dir {FOLDER}')
    print('total samples processed: ', max_num + 1)
    print('total negs passed: ', len(dfs))