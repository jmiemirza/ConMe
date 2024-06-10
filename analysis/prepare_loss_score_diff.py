



import pandas as pd

import os.path as osp
import json
import ast
import os
import random
import numpy
import matplotlib.pyplot as plt


def plot_two_histo(ours, orig, partition_name, model_name ):
    # x = [random.gauss(3, 1) for _ in range(400)]
    # y = [random.gauss(4, 2) for _ in range(400)]

    fig = plt.figure()
    bins = numpy.linspace(-2.5, 2.5, 100)

    plt.hist(ours, bins, alpha=0.5, label='ours')
    plt.hist(orig, bins, alpha=0.5, label='baseline')
    plt.title( f'{model_name} {partition_name}  diff = score_pos - score_neg' )
    plt.legend(loc='upper right')
    # pyplot.show()
    fig.savefig(f'/data1/lin/data/LMM_benchmarks/CombinedLLM/visual/ours_vs_baseline_{partition_name}.pdf')


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
if __name__ == '__main__':
    model_name = 'instructblip_flan_t5'
    if model_name == 'instructblip_flan_t5':
        # model_name_v2 = 'instructblip-flan-t5'
        model_name_v2 =  model_name

    # partition_name = 'replace_att'
    # dataset_name = 'sugar'

    partition_name = 'vga'
    dataset_name = 'vga'

    if partition_name == 'vga':
        score_diff_key = 'scores_diff_0'
        correct_caption_key = 'correct_option_0'
    else:
        score_diff_key = 'scores_diff'
        correct_caption_key = 'correct_option'

    our_results_file = osp.join( '/data1/lin/data/LMM_benchmarks/CombinedLLM', model_name, f'{partition_name}_{model_name}_combined_records.csv' )

    original_results_file = osp.join('/data1/lin/data/LMM_benchmarks/CombinedLLM/original_dataset', f'orig_{dataset_name}_{model_name_v2}.csv' )

    our_results = pd.read_csv(our_results_file)
    results_orig = pd.read_csv(original_results_file)

    n_samples = len(our_results['image'])
    our_results_diff_dict = {}
    our_results_diff_list = []
    for idx in range( n_samples):
        # our_results_diff_dict.update({our_results['image'][idx] :  our_results['scores_diff'][idx]}) #  score difference = score_neg - score_pos ,   if the score difference < 0,  classification is correct,  the lower the beter
        our_results_diff_dict.update( { our_results[correct_caption_key][idx]:  - our_results[score_diff_key][idx]   } )
        our_results_diff_list.append( - our_results[score_diff_key][idx])



    n_samples_in_orig = 0
    orig_results_diff_dict = dict()
    orig_results_diff_list = []
    for idx in range(len(results_orig['image'])):
        if results_orig['class'][idx] == partition_name:

            # scores_dict = json.loads( our_results_orig['scores'][idx])
            scores_dict = ast.literal_eval(results_orig['scores'][idx])
            keys = list(scores_dict.keys())
            true_caption = results_orig['correct_option'][idx]
            if keys[0] == true_caption:
                score_pos = scores_dict[keys[0]]
                score_neg = scores_dict[keys[1]]
            else:
                score_pos = scores_dict[keys[1]]
                score_neg = scores_dict[keys[0]]

            # orig_results_diff_dict.update( {our_results_orig['image'][idx] :  score_neg - score_pos  } )   #  score difference = score_neg - score_pos ,   if the score difference < 0,  classification is correct,  the lower the beter
            if results_orig['correct_option'][idx] in our_results_diff_dict.keys():
                n_samples_in_orig += 1
                orig_results_diff_dict.update({results_orig['correct_option'][idx]: - (score_neg - score_pos)})
                orig_results_diff_list.append( - ( score_neg - score_pos) )   #  score difference = score_neg - score_pos ,   if the score difference < 0,  classification is correct,  the lower the beter
                t =0
    plot_two_histo(ours=our_results_diff_list, orig=orig_results_diff_list, partition_name=partition_name, model_name=model_name)
    t = 1

