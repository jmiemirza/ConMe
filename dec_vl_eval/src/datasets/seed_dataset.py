import os
import json
from huggingface_hub import hf_hub_download
from zipfile import ZipFile

def get_seed_image_data(seed_data_root, seed_cc3m_root):
    # seed_data_root = "/dccstor/leonidka1/irenespace/data/seed"

    # process json file for dataset
    seed_json_path = f"{seed_data_root}/SEED-Bench.json"
    print(os.path.exists(seed_json_path))  # for debugging
    if not os.path.exists(seed_json_path):
        print("json for SEED-Bench could not be found!")
        try:
            hf_hub_download(
                repo_id="AILab-CVC/SEED-Bench",
                filename="SEED-Bench.json",
                repo_type='dataset',
                local_dir=seed_data_root,
                cache_dir='/dccstor/leonidka1/irenespace/.cache'
            )
            print('Successfully downloaded SEED-Bench json.')  # for debugging
        except:
            raise RuntimeError(
                "Unable to download SEED-Bench json.")

    seed_json = json.load(open(seed_json_path, 'rb'))  # double start matches directories
    if "questions" in seed_json:
        seed_json = seed_json['questions']  # list of questions

    # process image files for dataset
    # seed_cc3m_root = f"{seed_data_root}/seed_bench_image"
    if not os.path.exists(seed_cc3m_root):
        print("Image Directory for SEED-Bench could not be found!")
        try:
            hf_hub_download(
                repo_id="AILab-CVC/SEED-Bench",
                filename="SEED-Bench-image.zip",
                repo_type='dataset',
                local_dir=seed_cc3m_root,
                cache_dir='/dccstor/leonidka1/irenespace/.cache'
            )

            # unzip files
            with ZipFile(f"{seed_cc3m_root}/SEED-Bench-image.zip", 'r') as zObject:
                zObject.extractall(path=seed_cc3m_root)

            print('Successfully downloaded SEED-Bench images.')  # for debugging
        except:
            raise RuntimeError(
                "Unable to download SEED-Bench images.")

    return seed_json

