
inference_mode=loss_score
model=blip2_flan_t5_xl
gpu_nr=7
keyword=combined


for partition in replace_att replace_obj replace_rel vga vgr

do
  CUDA_VISIBLE_DEVICES=${gpu_nr} python eval.py --model ${model} --eval_dataset COMBINED_LLM --select_by ${inference_mode} --backend hf --eval_base_dir /data2/lin/data/CVPR24_VLM_experiments/${keyword}_${partition}_${model}_${inference_mode}/ --experiment_name ${partition} --job_name combined_${partition}_${model} --dataset_partition ${partition}
done