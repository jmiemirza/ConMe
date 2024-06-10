
inference_mode=generate
model=llava-v1.5-7b
gpu_nr=4
keyword=leaveout
model_path=/data2/lin/data/Llava_weights/llava-v1.5-7b
temperature=0
max_new_tokens=128
prompt=11

for partition in replace_att replace_obj replace_rel vga vgr

do
  CUDA_VISIBLE_DEVICES=${gpu_nr} python eval.py --model ${model} --eval_dataset COMBINED_LLM --select_by ${inference_mode} --backend hf --model_path ${model_path} --temperature ${temperature} --max_new_tokens ${max_new_tokens} --prompt ${prompt}   --eval_base_dir /data2/lin/data/CVPR24_VLM_experiments/${keyword}_${partition}_${model}_${inference_mode}/ --experiment_name ${partition} --job_name combined_${partition}_${model} --dataset_partition ${partition}

done
