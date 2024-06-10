import argparse


def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_name', default='debug')
    parser.add_argument('--job_name', default=None)
    parser.add_argument('--prompt_config', default='multiple_choice')
    parser.add_argument('--language_model', default='vicuna_7B')
    parser.add_argument('--dataset_id', default='vgr')
    parser.add_argument('--llm_temperature', default=1.0)
    parser.add_argument('--llm_max_new_tokens', default=300)
    parser.add_argument('--min_length', default=2)
    parser.add_argument('--length_penalty', default=1.1)
    parser.add_argument('--repetition_penalty', default=1.0)
    parser.add_argument('--answer_order', default='random')
    parser.add_argument('--n_shot', default=0)
    parser.add_argument('--n_blip_captions', default=0)
    parser.add_argument('--max_n_sentences', default=5)
    parser.add_argument('--seed', default=1989)
    parser.add_argument('--max_neg_flips', default=2)
    parser.add_argument('--num_samples', default=0)
    parser.add_argument('--use_ram_features', action='store_true')
    parser.add_argument('--combine_instances', action='store_true')
    parser.add_argument('--use_substrings', action='store_true')
    parser.add_argument('--use_only_captions', action='store_true')

    args = parser.parse_args()

    return args
