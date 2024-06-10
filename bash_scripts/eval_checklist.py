import random
import os
import subprocess
from collections import OrderedDict
from cvar_pyutils.ccc import submit_dependant_jobs

# enumerated params
params_enum = OrderedDict()
params_enum['experiment_name'] = ['exp', 'rest_of_benchmarks']
params_enum['model'] = ['mod', 'vicuna_7b']
params_enum['select_by'] = ['scor', 'generate', 'loss_score']
params_enum['dataset'] = ['data','ARO_VGA', 'COCO', "Flickr", "CREPE"]
# params_enum['caption_dataset'] = [
#     'mod',
#     None,
#     'LN_HF_Dataset.csv',
#     'Relations_LN_HF_Dataset.csv',
#     'LeftRight_LN_HF_Dataset.csv',
#     'Objs_and_Rels_LN_HF_Dataset.csv',
#     'LeftRight_v2_LN_HF_Dataset.csv',
#     'Relations_v2_LN_HF_Dataset.csv',
# ]

scratch_root = f"/dccstor/leonidka1/victor_space/scratch/dec_results/{params_enum['experiment_name'][-1]}"

# For debugging purposes
print_only = False

# fixed system params
is_multi_node = True
base_port = 10000
duration = '24h'
num_cores = 4 #12
num_workers = 2
num_nodes = 1
num_gpus = 1 #8
number_of_jobs = 1 #5
mem = '40g'
gpu_type = 'a100' # && hname!=cccxc451 && hname!=cccxc441'

# Define the command to run
command = f'python /dccstor/leonidka1/victor_space/dec-vl-eval/eval.py'

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
        for v in vals[1:]:
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
        return [ (cmd, name[1:]) ]

# # generate the command list
run_cmds = genCommands(params_enum, command, '', '')
print(len([print(x) for x in run_cmds])) # for debug

# get current job listing to prevent duplicates
job_listing_command="{ jbinfo -ll -state r; jbinfo -ll -state p;} | grep 'Job Name' | awk -F '>' '{print $2}' | awk -F '<' '{print $2}' | sort | uniq"
existing_jobs = subprocess.check_output(job_listing_command, shell=True).decode("utf-8").split('\n')
existing_jobs = set([x.strip() for x in existing_jobs if len(x.strip()) > 0])

# # run the jobs
for cmd in run_cmds:
    jcmd = cmd[0]
    jname = cmd[1]

    # For setting the logging correctly
    jcmd += " --job_name " + jname

    if jname in existing_jobs:
        print(f'=> skipping {jname} it is already running or pending...')
    else:
        print(f'{jname} - {jcmd}\n')
        if not print_only:
            result_dir = f'{scratch_root}/{jname}'

            # If the args.result_dir does not exist, create it.
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            submit_dependant_jobs(
                command_to_run=jcmd,
                name=jname,
                mem=mem,
                num_cores=num_cores,
                num_gpus=num_gpus,
                num_nodes=num_nodes,
                out_file=f"{result_dir}/job.out",
                err_file=f"{result_dir}/job.err",
                duration=duration,
                number_of_rolling_jobs=number_of_jobs,
                gpu_type=gpu_type
            )
