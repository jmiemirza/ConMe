import pickle
import os.path as osp


if __name__ == '__main__':
    file_path = '/data2/lin/data/SugarCrepe_experiments/debug/20231103_145216_instructblib_vicuna7b_loss_score_replacerel'
    file_path = osp.join(file_path, 'results_dataframe.pkl')

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    t = 1