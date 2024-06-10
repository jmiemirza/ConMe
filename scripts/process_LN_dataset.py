import sys
# Get the parent directory path
parent_directory = "/dccstor/leonidka1/victor_space/dec-vl-eval"# Add the parent directory to the system path
sys.path.append(parent_directory)
from finetune_dataset import LocalizedObjects
from utils import get_args
from cvar_pyutils.debugging_tools import set_remote_debugger

if __name__=="__main__":
    # args
    args = get_args()

    # setup debug
    if args.debug:
        set_remote_debugger(None, args.debug_port)

    ln_dataset = LocalizedObjects(
        dataset_size=args.dataset_size,
        write_dataset=args.write_dataset,
        datapoint_type=args.datapoint_type,
        csv_name=args.dataset_csv_path
    )
