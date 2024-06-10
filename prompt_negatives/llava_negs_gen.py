import os
import fire
from genai.credentials import Credentials
from genai.model import Model
from genai.schemas import GenerateParams

import json
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm
import re

prompt_template = 'Generate a sequence of contradictory sentences differing by one word and sharing a prefix with the original sentence.\n\
\n\
Input: original: A brown fox jumped over a fence into the river.\n\
Output: prefix 1: A brown fox\n\
contradiction 1: A brown dog\n\
prefix 2: A brown fox jumped\n\
contradiction 2: A brown fox crawled\n\
prefix 3: A brown fox jumped over\n\
contradiction 3: A brown fox jumped under\n\
prefix 4: A brown fox jumped over a fence\n\
contradiction 4: A brown fox jumped over a table\n\
prefix 5: A brown fox jumped over a fence into\n\
contradiction 5: A brown fox jumped over a fence above\n\
prefix 6: A brown fox jumped over a fence into the river\n\
contradiction 6: A brown fox jumped over a fence into the grass\n\
\n\
Input: original: A large truck stopped on the side of the road and a man in green shirt came out of it.\n\
Output: prefix 1: A large truck\n\
contradiction 1: A large horse\n\
prefix 2: A large truck stopped\n\
contradiction 2: A large truck drove\n\
prefix 3: A large truck stopped on the side\n\
contradiction 3: A large truck stopped on the middle\n\
prefix 4: A large truck stopped on the side of the road\n\
contradiction 4: A large truck stopped on the side of the park\n\
prefix 5: A large truck stopped on the side of the road and a man\n\
contradiction 5: A large truck stopped on the side of the road and a woman\n\
prefix 6: A large truck stopped on the side of the road and a man in green\n\
contradiction 6: A large truck stopped on the side of the road and a man in black\n\
prefix 7: A large truck stopped on the side of the road and a man in green shirt\n\
contradiction 7: A large truck stopped on the side of the road and a man in green pants\n\
prefix 8: A large truck stopped on the side of the road and a man in green shirt came\n\
contradiction 8: A large truck stopped on the side of the road and a man in green shirt jumped\n\
prefix 9: A large truck stopped on the side of the road and a man in green shirt came out\n\
contradiction 9: A large truck stopped on the side of the road and a man in green shirt came into\n\
\n\
Input: original: The cat is sleeping on the couch, but it will wake up soon.\n\
Output: prefix 1: The cat\n\
contradiction 1: The dog\n\
prefix 2: The cat is sleeping\n\
contradiction 2: The cat is awake\n\
prefix 3: The cat is sleeping on the couch\n\
contradiction 3: The cat is sleeping on the bed\n\
prefix 4: The cat is sleeping on the couch, but it will wake\n\
contradiction 4: The cat is sleeping on the couch, but it will sleep\n\
prefix 5: The cat is sleeping on the couch, but it will wake up soon\n\
contradiction 5: The cat is sleeping on the couch, but it will wake up later\n\
\n\
Input: original: <input>\n\
Output:'

def main(
    api_key=os.getenv("GENAI_KEY", 'pak-2Lc34c-Zf949eQ21xWwpRmN5RyVmdoZWXM3qG-gn8Bk'), # pak-2Lc34c-Zf949eQ21xWwpRmN5RyVmdoZWXM3qG-gn8Bk, pak-_yhHh8IcezgJdW2XcuzU6zH8EqhffxMXucUUbcW2B9E
    api_url = os.getenv("GENAI_API", "https://bam-api.res.ibm.com/v1"),
    src_data = 'data/blip_laion_cc_sbu_558k.json',
    out_path = 'data/negs',
    decoding_method = 'greedy',
    max_new_tokens = 600,
    model_name = 'meta-llama/llama-2-70b-chat',
    debug = False,
    debug_port = 12345
):
    if debug:
        from cvar_pyutils.debugging_tools import set_remote_debugger
        set_remote_debugger(None, debug_port)

    seed = int(int(datetime.now().timestamp() * 1000000)) % (2 ** 32 - 1)
    print(f'Setting random seed: {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    with open(src_data, 'r') as f:
        data = json.load(f)

    # make order unique
    np.random.shuffle(data)

    output_dir = os.path.join(out_path, os.path.basename(src_data).split('.')[0])
    os.makedirs(output_dir,exist_ok=True)

    creds = Credentials(api_key, api_endpoint=api_url)

    params = GenerateParams(
                decoding_method=decoding_method,
                max_new_tokens=max_new_tokens
            )

    generator = Model(model_name, params=params, credentials=creds)

    for d in tqdm(data):
        out_path = os.path.join(output_dir, str(d['id']) + '.json')
        if os.path.exists(out_path):
            continue
        for c in d['conversations']:
            if c['from'] == 'gpt':
                v = c['value']
                prompt = prompt_template.replace('<input>', v)
                for result in generator.generate_async([prompt]):
                    if result is not None:
                        res = result.generated_text
                        prevIx = 0
                        negs = []
                        for mtch in re.finditer('prefix [0-9]*:[^\n]*\ncontradiction [0-9]*:[^\n]*', res):
                            try:
                                txt = res[mtch.start():mtch.end()]
                                txt = txt.split('\n')
                                pref = txt[0].split(':')
                                cont = txt[1].split(':')
                                ixPref = int(pref[0].split(' ')[1])
                                ixCont = int(cont[0].split(' ')[1])
                                pref = pref[1].strip()
                                cont = cont[1].strip()
                                if not ((ixPref == ixCont) and (ixPref == prevIx + 1)):
                                    break
                                prevIx = ixPref
                                wP = pref.split(' ')
                                wC = cont.split(' ')
                                e = np.equal(wP, wC)
                                assert np.all(e[:-1]) and (not e[-1])
                                negs.append(cont)
                            except:
                                pass
                        c['negs'] = negs
        # store the json
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(d, f)


if __name__ == "__main__":
    fire.Fire(main)
