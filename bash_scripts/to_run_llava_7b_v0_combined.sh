
inference_mode=loss_score
model=llava-7b-v0
gpu_nr=7
keyword=combined


for partition in replace_att replace_obj replace_rel vga vgr

do
  CUDA_VISIBLE_DEVICES=${gpu_nr} python eval.py --model ${model} --eval_dataset COMBINED_LLM --select_by ${inference_mode} --backend hf --model_path /data1/lin/data/LLM_weights/Llava_weights/LLaVA-7B-v0/ --temperature 0 --max_new_tokens 128 --prompt 11 --eval_base_dir /data2/lin/data/CVPR24_VLM_experiments/${keyword}_${partition}_${model}_${inference_mode}/ --experiment_name ${partition} --job_name combined_${partition}_${model} --dataset_partition ${partition}

done