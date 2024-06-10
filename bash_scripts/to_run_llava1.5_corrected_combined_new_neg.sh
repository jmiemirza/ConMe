
inference_mode=loss_score
model=llava-v1.5-7b
gpu_nr=5
dataset_name=COMBINED_LLM
keyword=combined
model_path=/system/user/publicdata/llm/Llava_weights/Llava_weights/llava-v1.5-7b
result_main_dir=/system/user/publicdata/LMM_benchmarks/CVPR24_VLM_experiments

which_neg=new_neg

experiment_name=${keyword}_${model}_${inference_mode}_${which_neg}_corrected_w_template

for partition in replace_att replace_obj replace_rel vga vgr

do
  CUDA_VISIBLE_DEVICES=${gpu_nr} python eval.py --model ${model} \
  --eval_dataset $dataset_name \
  --select_by ${inference_mode} \
  --backend hf \
  --model_path $model_path --temperature 0 --max_new_tokens 128 --prompt 11 \
  --eval_base_dir ${result_main_dir} \
  --experiment_name $experiment_name \
  --job_name $partition \
  --dataset_partition ${partition} \
  --which_neg $which_neg

done