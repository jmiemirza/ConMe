
inference_mode=loss_score
model=llava-v1.5-7b
gpu_nr=5

for i in "sugar_baseline SUGAR"
do
  set -- $i # Convert the "tuple" into the param args $1 $2...
  keyword=$1
  eval_dataset=$2
  partition=None
  CUDA_VISIBLE_DEVICES=${gpu_nr} python eval.py --model ${model} --eval_dataset ${eval_dataset} --select_by ${inference_mode} --backend hf --model_path /data1/lin/data/LLM_weights/Llava_weights/LLaVA-7B-v0/ --temperature 0 --max_new_tokens 128 --prompt 11 --eval_base_dir /data2/lin/data/CVPR24_VLM_experiments/${keyword}_${partition}_${model}_${inference_mode}_corrected_v2/ --experiment_name ${partition} --job_name ${partition}_${model} --dataset_partition ${partition}

done