
inference_mode=loss_score
model=instructblip_vicuna_7b
gpu_nr=7

for i in "aro_vgr_baseline ARO_VGR" "aro_vga_baseline ARO_VGA"
do
  set -- $i # Convert the "tuple" into the param args $1 $2...
  keyword=$1
  eval_dataset=$2
  partition=None
  CUDA_VISIBLE_DEVICES=${gpu_nr} python eval.py --model ${model} --eval_dataset ${eval_dataset} --select_by ${inference_mode} --backend hf --eval_base_dir /system/user/publicdata/LMM_benchmarks/CVPR24_VLM_experiments/${keyword}_${partition}_${model}_${inference_mode}_textonly/ --experiment_name ${partition} --job_name ${partition}_${model} --dataset_partition ${partition}

done