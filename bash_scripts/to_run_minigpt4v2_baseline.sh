
inference_mode=loss_score
model=minigpt4v2
gpu_nr=7
keyword=aro_vgr_baseline

eval_dataset=ARO_VGR

for partition in None

do
  CUDA_VISIBLE_DEVICES=${gpu_nr} python eval.py --model ${model} --eval_dataset ${eval_dataset} --select_by ${inference_mode} --backend hf --eval_base_dir /data2/lin/data/CVPR24_VLM_experiments/${keyword}_${partition}_${model}_${inference_mode}/ --experiment_name ${partition} --job_name ${partition}_${model} --dataset_partition ${partition}

done