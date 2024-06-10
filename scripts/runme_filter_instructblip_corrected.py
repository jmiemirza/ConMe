import random
import subprocess
from collections import OrderedDict
from cvar_pyutils.ccc import submit_dependant_jobs
# import torch
import os
# from utils import get_args
import json
import time

# import os
# os.environ['NCCL_DEBUG']='INFO'

# enumerated params
params_enum = OrderedDict() # order is preserved, ! in the end means the field is mandatory in naming, will allways be included in the name
# m goes to job and folder name; right side can be a list of options
# params_enum['model'] = ['m', 'llava-v1.5-7b'] #sandstone_flan_generic_basewatson_v2 # 'google/flan-t5-base' 'google/flan-t5-large' 'google/flan-t5-xl'
params_enum['eval_dataset'] = ['ed', 'SAMPLE_NEG_UPDATED'] # llava_mix
params_enum['eval_vlm'] = ['ev', 'instructblip_flan_t5']
# params_enum['filter_vlm'] = ['fv', 'instructblip_flan_t5']
params_enum['dataset_partition'] = ['dp', 'replace_obj']
# params_enum['select_by'] = ['s', 'generate'] # generate or loss_score
# params_enum['num_samples'] = ['ns', 100]

# for llava-v1.5-7b
# params_enum['temperature'] = ['t', 0]
# params_enum['max_new_tokens'] = ['nt', 128]
# params_enum['prompt'] = ['p', 11]

# params_enum['bf16'] = ['bf16', True]

# general control params / flags
rerun_prefix = '' # add string to all runs
print_only = False
eval_only = False
summarize = False
summarize_launch_missing = False

# specific method general params
output_root = '/dccstor/leonidka1/irenespace/llm_results_updated/filter/instructblip_flan_t5/mistral'
# output_root = '/dccstor/leonidka1/irenespace/llm_results_updated/filter/instructblip_vicuna_7b/'

# fixed system params
is_multi_node = True
base_port = 10000
duration = '6h'
num_cores = 4
# num_workers = 4
num_nodes = 1 #if eval_only else 4
num_gpus = 1 if eval_only else 1
number_of_jobs = 1
mem = '80g'
gpu_type = 'a100' # && hname!=cccxc451 && hname!=cccxc441'
# 'a100_80gb' for bigger gpu

# multi-node
# if is_multi_node:
#     # if int(torch.__version__.split('.')[1]) < 9:
#     #     print(f'Detected torch sub-version below 9, using torch.launch')
#     #     cmd_prefix = f"pyutils-launch"
#     # else:
#     print(f'Detected torch sub-version 9 or above, using torch.run')
#     cmd_prefix = f"pyutils-run" #f"pyutils-accelerate" #f"pyutils-run"
# else:
#     cmd_prefix = "python"

cmd_prefix = "python"

# command related
# command = f'export TRANSFORMERS_CACHE=/dccstor/leonidka1/irenespace/.cache; {cmd_prefix}  eval.py --experiment_name baseline --job_name sugar_llava-v1.5-7b'
# command = f'export TRANSFORMERS_CACHE=/dccstor/leonidka1/irenespace/.cache; {cmd_prefix} -m prompt_negatives.filter_llava_mix'
command = f'export TRANSFORMERS_CACHE=/dccstor/leonidka1/irenespace/.cache; {cmd_prefix} -m prompt_negatives.filter_negs_updated'


# for recursion
#moco_edges_{edges}_i2i_{i2i}_e2e_{e2e}_e2i_{e2i}_sb_{split_backbones}
def genCommands( params_enum, cmd, name, exp_name ):
    params_enum = params_enum.copy()
    if len(params_enum) > 0:
        cmds = []
        key = list(params_enum.keys())[0]
        vals = params_enum[key]
        en = vals[0]
        mandatory = False
        if en[-1] == '!':
            mandatory = True
            en = en[:-1]
        params_enum.pop(key)
        sn = en if len(en) < len(key) else key
        for iV, v in enumerate(vals[1:]):
            if isinstance(v, bool):
                sub_cmd = f' --{key}' if v else ''
                sub_name = f':{sn}' if v else ''
                sub_exp_name = f'{en}_{{{key}}}_' if v or mandatory else ''
            else:
                sub_cmd = f' --{key} {str(v)}' if v is not None else ''
                sub_name = f':{sn}={str(v)}' if v is not None else ''
                sub_exp_name = f'{en}_{str(v)}_' if v is not None else ''
            cmds.extend(genCommands(params_enum, cmd + sub_cmd, name + sub_name, exp_name + sub_exp_name))
        return cmds
    else:
        port = base_port + random.randint(1, 10000)
        cmd = cmd.format(port=port)
        # cmd += f' --eval_base_dir {output_root}/{rerun_prefix}{exp_name[:-1]}'
        cmd += f' --output_path {output_root}/{rerun_prefix}{exp_name[:-1]}'
        return [ (cmd, rerun_prefix + name[1:]) ]

# generate the command list
run_cmds = genCommands(params_enum, command, '', '')
# print(len([print(x) for x in run_cmds])) # for debug

# get current job listing to prevent duplicates
job_listing_command="{ jbinfo -ll -state r; jbinfo -ll -state p;} | grep 'Job Name' | awk -F '>' '{print $2}' | awk -F '<' '{print $2}' | sort | uniq"
existing_jobs = subprocess.check_output(job_listing_command, shell=True).decode("utf-8").split('\n')
existing_jobs = set([x.strip() for x in existing_jobs if len(x.strip()) > 0])

# # run the jobs
jlog_root = '_job_logs'
res_summary = {}
for cmd in run_cmds:
    jcmd = cmd[0]
    jname = cmd[1]
    jname = jname.replace('/', '_')

    os.makedirs(jlog_root, exist_ok=True)

    to_launch = True
    # if summarize:
    #     # from cvar_pyutils.debugging_tools import set_remote_debugger
    #     # set_remote_debugger(None, 12345)
    #     cmd_params = (jcmd.split('main.py ')[1]).split(' ')
    #     cur_args = get_args(cmd_params)
    #     res_path = os.path.join(cur_args.output_dir, 'results.txt')
    #     group_name = jname.split('_uid=')
    #     user_id = int(group_name[1])
    #     group_name = group_name[0]
    #     if os.path.exists(res_path):
    #         with open(res_path, 'r') as f:
    #             lines = f.readlines()
    #             res = lines[0].split('Rouge: ')[1].strip().replace("'",'"')
    #         res = json.loads(res)
    #         if group_name in res_summary:
    #             prev_res, cnt = res_summary[group_name]
    #             for k in prev_res:
    #                 prev_res[k] = (prev_res[k] * cnt + res[k]) / (cnt + 1)
    #             cnt += 1
    #             res_summary[group_name] = (prev_res, cnt)
    #         else:
    #             res_summary[group_name] = (res, 1)
    #     else:
    #         print(f'Missing group: {group_name} user {user_id}')
    #         if summarize_launch_missing:
    #             to_launch = True
    # else:
    #     to_launch = True

    if to_launch:
        if (not print_only) and (jname in existing_jobs):
            print(f'=> skipping {jname} it is already running or pending...')
        else:
            print(f'{jname} - {jcmd}\n')
            if not print_only:
                submit_dependant_jobs(
                    command_to_run=jcmd,
                    name=jname,
                    mem=mem, num_cores=num_cores, num_gpus=num_gpus, num_nodes=num_nodes,
                    duration=duration, number_of_rolling_jobs=number_of_jobs,
                    gpu_type=gpu_type,
                    out_file=os.path.join(jlog_root, f'{jname}_out.log'),
                    err_file=os.path.join(jlog_root, f'{jname}_err.log')
                )

if summarize:
    for k in res_summary:
        print(f'{k}: {res_summary[k]}')