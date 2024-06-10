
inference_mode=loss_score
model=instructblip_vicuna_7b
gpu_nr=7
keyword=combined


for partition in vga vgr

do
  CUDA_VISIBLE_DEVICES=${gpu_nr} python eval.py --model ${model} --eval_dataset COMBINED_LLM --select_by ${inference_mode} --backend hf --eval_base_dir /system/user/publicdata/LMM_benchmarks/CVPR24_VLM_experiments/${keyword}_${partition}_${model}_${inference_mode}_baseline_corrected/ --experiment_name ${partition} --job_name combined_${partition}_${model} --dataset_partition ${partition}

done