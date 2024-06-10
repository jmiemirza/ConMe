import numpy as np
import pandas as pd
import json
import argparse
import os
import ast

NEG_SAMPLE_FILES = {
    'vga': '/dccstor/leonidka1/irenespace/scratch/prompt_negatives/aro/20231112_0035/falcon-180b_vga_all-samples.csv',
    'vgr': '/dccstor/leonidka1/irenespace/scratch/prompt_negatives/aro/20231112_1356/falcon-180b_vgr_all-samples.csv',
    'replace_obj': '/dccstor/leonidka1/irenespace/scratch/prompt_negatives/sugar/20231105_2219/falcon-180b_replace_obj_all-samples.csv',
    'replace_att': "/dccstor/leonidka1/irenespace/scratch/prompt_negatives/sugar/20231105_1456/falcon-180b_replace_att_all-samples.csv",
    'replace_rel': "/dccstor/leonidka1/irenespace/scratch/prompt_negatives/sugar/20231111_1453/falcon-180b_replace_rel_all-samples.csv",
}

NEG_CHANGE_WORD = {
    'vga': 'attribute',
    'vgr': 'preposition',
    'replace_obj': 'object',
    'replace_att': 'attribute',
    'replace_rel': 'preposition',
}

# for each model/partition result
def combine_files(args):

    # get results df and sort by original_id
    results_df = pd.read_csv(
        f'/dccstor/leonidka1/irenespace/scratch/prompt_negatives/{args.dataset_dir}/results/{args.dataset_partition}_{args.model}/{args.dataset_partition}_{args.model}_results.csv'
    ).sort_values(
        by=['original_id'], ascending=True
    )

    # # get score diffs df and sort by orig_id
    # diffs_df = pd.read_csv(
    #         f'/dccstor/leonidka1/irenespace/scratch/prompt_negatives/{args.dataset_dir}/results/{args.dataset_partition}_{args.model}/score_diffs.csv'
    #     ).sort_values(
    #         by=['orig_id'], ascending=True
    #     )

    # get all samples df and sort by orig_ind, drop second duplicates
    samples_df = pd.read_csv(
        NEG_SAMPLE_FILES[args.dataset_partition]
    ).sort_values(
        by=['orig_ind'], ascending=True
    ).drop_duplicates(subset=['orig_ind'])

    combined_records = []

    # for each row in results dataframe,
    for ind, row in results_df.iterrows():

        # use json to parse the scores dictionary
        try:
            correct_option, orig_id = row['correct_option'], row['original_id']
            result_scores = row['scores']

        except KeyError:
            correct_option, orig_id = row['correct_option_0'], row['original_id']
            result_scores = row['scores_0']

        except:
            raise ValueError("Unable to get correct option or scores from results file.")

        print(f'results ind {ind} and original ind {orig_id}')

        result_scores = ast.literal_eval(result_scores)
        print(result_scores)

        # using the correct option, identify the original positive and LLM neg
        gen_neg = [k for k in result_scores.keys() if k != correct_option][0]

        # compare to each of the new_neg_{ind} in score diffs until identify the matching one
        neg_ind = None
        samples_row = samples_df[samples_df['orig_ind'] == orig_id].iloc[0]
        for sample_ind in range(5):
            if samples_row[f'new_neg_{sample_ind}'] == gen_neg:
                neg_ind = sample_ind
                break

        assert samples_row['orig_ind'] == orig_id
        assert samples_row['image_path'] == row['image']
        assert neg_ind is not None

        print('-' * 50)
        # append acc, class, scores, scores_diff, orig_id, image_path, orig_pos, orig_neg, new_neg, new_change, new_contradiction,
        # with model, backend, prompt_id, benchmark, select_by
        if not isinstance(samples_row[f'new_change_{neg_ind}'], str):
            selected_change = ''
        else:
            selected_change = samples_row[f'new_change_{neg_ind}'].split(f'Change of {NEG_CHANGE_WORD[args.dataset_partition]}:')[-1].strip()

        item = {
            'orig_ind': samples_row['orig_ind'],
            'selected_neg_ind': neg_ind,
            'orig_pos': samples_row[f'orig_pos'],
            'orig_neg': samples_row[f'orig_neg'],
            'new_neg': samples_row[f'new_neg_{neg_ind}'],
            'new_change': selected_change,
            'contradiction_check': samples_row[f'new_contradict_{neg_ind}'],
        }

        item.update({**row})
        combined_records.append(item)

    output_file = f'/dccstor/leonidka1/irenespace/scratch/prompt_negatives/{args.dataset_dir}/results/{args.dataset_partition}_{args.model}/{args.dataset_partition}_{args.model}_combined_records.csv'
    pd.DataFrame(combined_records).to_csv(output_file)
    print(f'combined records saved to csv at {output_file}')

if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument("--dataset_dir",
                        default='sugar', # sugar or aro
                        type=str,
                        help="Which dataset to use for evaluation.")
    parser.add_argument("--dataset_partition",
                        default=None,
                        type=str,
                        help="Which partition of the dataset, if any.")

    # model args
    parser.add_argument("--model",
                        default='instructblip_flan_t5',
                        type=str,
                        help="Which pretrained model to evaluate/train from.")

    parser.add_argument('--debug',
                        action='store_true',
                        help='to debug')

    args = parser.parse_args()

    if args.debug:
        import pydevd_pycharm

        debug_ip = os.environ.get('SSH_CONNECTION', None)
        pydevd_pycharm.settrace(debug_ip, port=12345, stdoutToServer=True, stderrToServer=True, suspend=False)

    combine_files(args)