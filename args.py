import argparse
import os
from typing import Any
import pickle


def get_args(ext_args_str=None):

    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()

    # Job args
    parser.add_argument("--local-rank",
                        default=os.environ.get('LOCAL_RANK', 0),
                        type=int,
                        help="Please ignore and do not set this argument.")
    parser.add_argument('--debug',
                        action='store_true',
                        help='to debug')
    parser.add_argument("--debug_port",
                        default=12345,
                        type=int,
                        help="Please ignore and do not set this argument.")
    parser.add_argument("--num_workers",
                        default=0,
                        type=int,
                        help="Number of processes to spawn for getting individual batches.")
    parser.add_argument("--device",
                        default=0,
                        type=str,
                        help="Which GPU to put the model/data on by default.")
    parser.add_argument("--job_name",
                        default=None,
                        type=str,
                        help="Used to correspond to each evaluation/fine-tuning run.")
    parser.add_argument("--experiment_name",
                        default="debug",
                        type=str,
                        help="Used to group together multiple runs for evaluation/fine-tuning.")
    parser.add_argument("--eval",
                        default=False,
                        type=bool,
                        help='DEPRECATED. Used to just eval the peft model.')

    # model args
    parser.add_argument("--model",
                        default='instructblip_flan_t5',
                        type=str,
                        help="Which pretrained model to evaluate/train from.")
    parser.add_argument("--backend",
                        default='hf',
                        type=str,
                        help="The code-base of the model to use. Options: lavis and hf (huggingface)")

    # train
    parser.add_argument("--num_epochs",
                        default=100,
                        type=int,
                        help="Number of complete passes of training data to go through.")
    parser.add_argument("--label_smoothing_factor",
                        default=0.0,
                        type=float,
                        help="Amount of label smoothing to prevent over-confidence in one-hot labels.")
    parser.add_argument("--eval_steps",
                        default=100,
                        type=int,
                        help="Amount of training steps to take before running evaluation.")
    parser.add_argument("--init_lr",
                        default=0.0001,
                        type=float,
                        help="Initial learning rate.")
    parser.add_argument("--train_batch_sz",
                        default=4,
                        type=int,
                        help="Batch size of examples for fine-tuning.")
    parser.add_argument("--eval_batch_sz", default=4,
                        type=int,
                        help="Batch size during evaluation. NOTE: Huggingface stores each eval prediction on GPU.")
    parser.add_argument("--lora_r",
                        default=16,
                        type=int,
                        help="Rank of the LoRA fine-tuning parameterization.")
    parser.add_argument("--gradient_checkpointing",
                        default=False,
                        type=bool,
                        help="Whether or not to use gradient-checkpointing which can be helpful with big models.")
    parser.add_argument("--lora_target_modules",
                        default=['q', 'v'],
                        type=list,
                        help="")
    parser.add_argument("--finetune_dataset",
                        default="LN_HF_Dataset.csv",
                        type=str,
                        help="Which dataset the LoRA fine-tuning be done on. Refer to README for a data breakdown.")
    parser.add_argument("--train_base_dir",
                        default="/dccstor/leonidka1/victor_space/scratch/train_dec_results",
                        type=str,
                        help="Place to store the output files from the fine-tuning.")

    # Output
    parser.add_argument('--eval_base_dir',
                        default='/dccstor/leonidka1/irenespace/scratch/dec_results/debug',
                        help="Directory where files are saved for different evaluation runs.")

    # inference
    parser.add_argument("--save_every",
                        default=100,
                        type=int,
                        help="How often to save the dataframe of statistics per evaluation prediction.")
    parser.add_argument("--gen_type",
                        default='beam',
                        type=str,
                        help="What kind of generation scheme to use for outputting text.")
    parser.add_argument("--num_beams",
                        default=5,
                        type=int,
                        help="How many beams to use during beam search.")
    parser.add_argument("--eval_dataset",
                        default='SUGAR',
                        type=str,
                        help="What dataset to use to evaluate a Vision-Language model.")
    parser.add_argument("--filter_dataset",
                        default='SAMPLE_NEG_UPDATED',
                        type=str,
                        help="What type of dataset used during the filtering process.")
    parser.add_argument("--dataset_partition",
                        default=None,
                        type=str,
                        help="A specific partition of the dataset to evaluate on.")
    parser.add_argument("--prompt_model",
                        type=str,
                        default='llm',
                        help="Which type of model was used for prompting, either llm or vlm.")
    parser.add_argument("--select_by",
                        default='generate',
                        type=str,
                        help="How to score the output of the model, by either direct comparison or by loss score.")
    parser.add_argument("--num_samples",
                        default=None,
                        type=int,
                        help="Number of samples to take for evaluation, some datasets are quite large.")
    parser.add_argument("--num_neg_options",
                        type=int,
                        default=1,
                        help="For the eval_datasets which have multiple negative options, choose how many to offer.")
    parser.add_argument("--pretrained_root",
                        default='/dccstor/leonidka1/victor_space/scratch/train_dec_results/checkpoint_logs',
                        type=str,
                        help="Root where the fine-tuned caption models are stored to load from.")
    parser.add_argument("--caption_dataset",
                        default=None,
                        type=str,
                        help="This will select the best model from fine-tuning of this caption_dataset to be used "
                             "during eval.")
    parser.add_argument("--filter_vlm",
                        default='instructblip_flan_t5',
                        type=str,
                        help="Which vlm to use for selecting challenging negs, for filter_llava_mix dataset.")
    parser.add_argument("--which_neg", type=str, choices=['orig_neg', 'new_neg'])

    # gpt4v args
    parser.add_argument("--gpt4v_type",
                        default='dim', # dim or desc
                        type=str,
                        help="Type of generation for initial prompt in step 1 of gpt4v-generation pipeline.")
    parser.add_argument("--gpt4v_dim",
                        default=None,
                        type=str,
                        help="Specific dimension used in gpt4v-generation pipeline.")
    parser.add_argument("--gpt4v_dataset_step",
                        default=None,
                        type=int,
                        help="Which step the dataset was generated from in the gpt4v-generation pipeline.")

    # Controlling the prompt
    parser.add_argument("--prompt",
                        default='1',
                        type=str,
                        help="Different prompt templates to use to prompt the language model with.")
    parser.add_argument("--prompt_type",
                        default='option',
                        type=str,
                        help="Either offer the kind of prompt which offers options (option) or asks generic questions ("
                             "generic).")

    # dataset processing of Localized Narrative caption datasets
    parser.add_argument("--dataset_csv_path",
                        type=str,
                        help="Where to store the caption dataset.") # No default here because it's important to specify.
    parser.add_argument("--write_dataset",
                        default=False,
                        type=bool,
                        help="Whether or not to write the dataset or just look at the output (used for debugging).")
    parser.add_argument("--datapoint_type",
                        default="objects",
                        type=str,
                        help="Whether to use just the objects in the prompt or also include the relations.")
    parser.add_argument("--dataset_size",
                        default=-1,
                        type=int,
                        help="Determine the size of the fine-tuning dataset.")

    # Llava parameters
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model_base", type=str, default=None)
    # parser.add_argument("--image-file", type=str, required=True)
    # parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv_mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--load_8bit", action="store_true")
    parser.add_argument("--load_4bit", action="store_true")
    # parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image_aspect_ratio", type=str, default='pad')

    # MiniGPT4V2 parameters
    parser.add_argument("--cfg_path", type=str, default="./dec_vl_eval/minigpt4v2_eval_configs/minigptv2_eval.yaml")
    parser.add_argument("--options", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)


    if ext_args_str is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(ext_args_str)

    args.eval_base_dir = args.eval_base_dir.format(**args.__dict__)
    os.makedirs(args.eval_base_dir, exist_ok=True)
    if ext_args_str is None:
        print(f'Eval base dir: {args.eval_base_dir}')

    return args


def get_model_device(model):
    return model.parameters().__next__().device


def save_pickle(object: Any, filename: str):
    with open(filename, "wb") as fp:
        pickle.dump(object, fp)


def load_pickle(filename: str) -> Any:
    with open(filename, "rb") as fp:
        data = pickle.load(fp)
    return data

