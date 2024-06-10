import sys
import os
sys.path.append("..")  # last level
sys.path.append(os.path.abspath('../..'))

import argparse
import os.path as osp
from basics_eval_mme import main_eval_mme

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    # parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    args = parser.parse_args()

    args.model_path = '/data2/lin/data/Llava_weights/llava-v1.5-7b'
    args.model_base = None

    args.temperature = 0
    args.max_new_tokens = 128
    args.eval_result_dir = f'/data1/lin/data/Filter_llava_instruct_results/eval_MME/llava-1.5_7b_cli_inference_wo_period_temp{args.temperature}_max{args.max_new_tokens}'
    args.img_dir = '/data1/lin/data/llava_datasets/COCO_train2017/train2017'
    args.MME_dir = '/data1/lin/data/LMM_benchmarks/MME/MME_Benchmark_release_version'
    args.MME_img_format = '.jpg'

    args.result_template_dir = '/data1/lin/data/LMM_benchmarks/MME/eval_tool/Your_Results'


    main_eval_mme(args)

