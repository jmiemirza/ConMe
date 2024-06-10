
inference_mode=loss_score
model=blip2_flan_t5_xl
gpu_nr=6

for i in "aro_vgr_baseline ARO_VGR" "aro_vga_baseline ARO_VGA" "sugar_baseline SUGAR"
do
  set -- $i # Convert the "tuple" into the param args $1 $2...
  keyword=$1
  eval_dataset=$2
  partition=None
  CUDA_VISIBLE_DEVICES=${gpu_nr} python eval.py --model ${model} --eval_dataset ${eval_dataset} --select_by ${inference_mode} --backend hf --eval_base_dir /data2/lin/data/CVPR24_VLM_experiments/${keyword}_${partition}_${model}_${inference_mode}/ --experiment_name ${partition} --job_name ${partition}_${model} --dataset_partition ${partition}

done