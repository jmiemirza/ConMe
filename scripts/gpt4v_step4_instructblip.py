import random
import subprocess
from collections import OrderedDict
from cvar_pyutils.ccc import submit_dependant_jobs
import torch
import os
# from utils import get_args
import json
import time
from dotenv import load_dotenv

# import os
# os.environ['NCCL_DEBUG']='INFO'
load_dotenv()
MY_API_KEY = os.getenv("GENAI_KEY", None)

# enumerated params
params_enum = OrderedDict() # order is preserved, ! in the end means the field is mandatory in naming, will allways be included in the name
# params_enum['debug_port'] = ['jnum'] + [x for x in range(1)] #sandstone_flan_generic_basewatson_v2 # 'google/flan-t5-base' 'google/flan-t5-large' 'google/flan-t5-xl'
# params_enum['genai_key'] = ['gk', MY_API_KEY, "pak-2Lc34c-Zf949eQ21xWwpRmN5RyVmdoZWXM3qG-gn8Bk", "pak-_yhHh8IcezgJdW2XcuzU6zH8EqhffxMXucUUbcW2B9E", "pak-6KqEe6Q6P-kVUrWri6JGaQUEO1pOqcFL-Kijzv0hn0Q"]

# general control params / flags
rerun_prefix = ''
print_only = False
eval_only = False
summarize = False
summarize_launch_missing = False

# specific method general params
select_by = 'generate'
output_root = f'/dccstor/leonidka1/irenespace/gpt4v_results/desc/step4/'

# partition = "replace_rel"
# num_samples = 333
# experiment_name = f"{partition}_{num_samples}"
partition = "ln_finetune"
experiment_name = f"ln_finetune"

model = "instructblip_flan_t5"
# model = "instructblip_vicuna_7b"

# fixed system params
is_multi_node = False
base_port = 10000
duration = '12h'
num_cores = 4
# num_workers = 4
num_nodes = 1 #if eval_only else 4
num_gpus = 1 if eval_only else 1
number_of_jobs = 1
mem = '80g'
gpu_type = 'a100' # && hname!=cccxc451 && hname!=cccxc441'

# command related
command = f'export TRANSFORMERS_CACHE=/dccstor/leonidka1/irenespace/.cache; python -m prompt_negatives.vlm_eval_gpt4v --gpt4v_type desc --gpt4v_dataset_step 3 --eval_dataset GPT4V_EVAL --dataset_partition {partition} --select_by {select_by} --model {model} --experiment_name {experiment_name} --job_name {model}'

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
                sub_exp_name = f'{en}_{{{key}}}_' if v is not None else ''
            cmds.extend(genCommands(params_enum, cmd + sub_cmd, name + sub_name, exp_name + sub_exp_name))
        return cmds
    else:
        port = base_port + random.randint(1, 10000)
        cmd = cmd.format(port=port)
        cmd += f' --eval_base_dir {output_root}'
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

    to_launch = False
    if summarize:
        # from cvar_pyutils.debugging_tools import set_remote_debugger
        # set_remote_debugger(None, 12345)
        cmd_params = (jcmd.split('main.py ')[1]).split(' ')
        cur_args = get_args(cmd_params)
        res_path = os.path.join(cur_args.output_dir, 'results.txt')
        group_name = jname.split('_uid=')
        user_id = int(group_name[1])
        group_name = group_name[0]
        if os.path.exists(res_path):
            with open(res_path, 'r') as f:
                lines = f.readlines()
                res = lines[0].split('Rouge: ')[1].strip().replace("'",'"')
            res = json.loads(res)
            if group_name in res_summary:
                prev_res, cnt = res_summary[group_name]
                for k in prev_res:
                    prev_res[k] = (prev_res[k] * cnt + res[k]) / (cnt + 1)
                cnt += 1
                res_summary[group_name] = (prev_res, cnt)
            else:
                res_summary[group_name] = (res, 1)
        else:
            print(f'Missing group: {group_name} user {user_id}')
            if summarize_launch_missing:
                to_launch = True
    else:
        to_launch = True

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