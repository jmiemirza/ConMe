
#partition=replace_att
#CUDA_VISIBLE_DEVICES=7 python eval.py --model llava-v1.5-7b --eval_dataset COMBINED_LLM --select_by generate --backend hf --model_path /data2/lin/data/Llava_weights/llava-v1.5-7b --temperature 0 --max_new_tokens 128 --prompt 11 --eval_base_dir /data2/lin/data/CVPR24_VLM_experiments/combined_${partition}_llava1.5_generate/ --experiment_name ${partition} --job_name combined_${partition}_llava1.5 --dataset_partition ${partition}

partition=replace_obj
CUDA_VISIBLE_DEVICES=7 python eval.py --model llava-v1.5-7b --eval_dataset COMBINED_LLM --select_by generate --backend hf --model_path /data2/lin/data/Llava_weights/llava-v1.5-7b --temperature 0 --max_new_tokens 128 --prompt 11 --eval_base_dir /data2/lin/data/CVPR24_VLM_experiments/combined_${partition}_llava1.5_generate/ --experiment_name ${partition} --job_name combined_${partition}_llava1.5 --dataset_partition ${partition}

partition=replace_rel
CUDA_VISIBLE_DEVICES=7 python eval.py --model llava-v1.5-7b --eval_dataset COMBINED_LLM --select_by generate --backend hf --model_path /data2/lin/data/Llava_weights/llava-v1.5-7b --temperature 0 --max_new_tokens 128 --prompt 11 --eval_base_dir /data2/lin/data/CVPR24_VLM_experiments/combined_${partition}_llava1.5_generate/ --experiment_name ${partition} --job_name combined_${partition}_llava1.5 --dataset_partition ${partition}

partition=vga
CUDA_VISIBLE_DEVICES=7 python eval.py --model llava-v1.5-7b --eval_dataset COMBINED_LLM --select_by generate --backend hf --model_path /data2/lin/data/Llava_weights/llava-v1.5-7b --temperature 0 --max_new_tokens 128 --prompt 11 --eval_base_dir /data2/lin/data/CVPR24_VLM_experiments/combined_${partition}_llava1.5_generate/ --experiment_name ${partition} --job_name combined_${partition}_llava1.5 --dataset_partition ${partition}

partition=vgr
CUDA_VISIBLE_DEVICES=7 python eval.py --model llava-v1.5-7b --eval_dataset COMBINED_LLM --select_by generate --backend hf --model_path /data2/lin/data/Llava_weights/llava-v1.5-7b --temperature 0 --max_new_tokens 128 --prompt 11 --eval_base_dir /data2/lin/data/CVPR24_VLM_experiments/combined_${partition}_llava1.5_generate/ --experiment_name ${partition} --job_name combined_${partition}_llava1.5 --dataset_partition ${partition}


