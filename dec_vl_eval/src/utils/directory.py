# Options which involve options (A, B)
option_prompt_dict = {
    "1": 'Option A: <txt1>. Option B: <txt2>. Which option is correct for this image',
    "2": 'Option A: <txt1>. Option B: <txt2>. Described from the viewpoint of looking towards the image, which option is correct',
    "3": 'Option A: <txt1>. Option B: <txt2>. Described from the viewpoint of looking at the image, which option is correct',
    "4": 'Option A: <txt1>. Option B: <txt2>. From the camera viewpoint of looking towards the image, which option is correct',
    "5": 'Option A: <txt1>. Option B: <txt2>. Looking at the image, which option is correct',
    "6": 'Option A: <txt1>. Option B: <txt2>. Looking towards the image, which option is correct',
    "7": 'Option A: <txt1>. Option B: <txt2>. With the camera facing into the image plane, which option is correct',
    "8": 'Option A: <txt1>. Option B: <txt2>. With the image in front of you, which option is correct',
    "9": 'Option A: <txt1>. Option B: <txt2>. Which relation describes the image',
    "10": 'Option A: <txt1>. Option B: <txt2>. With the image in front of you, which option is correct',
    "11": 'Option A: <txt1>. Option B: <txt2>. Which option is correct for this image? Answer with the option’s letter from the given choices directly.',
}

# Generic prompts which don't involve options
generic_prompt_dict = {
    "1": 'What is in the image',
    "2": 'Describe the things in the image',
    "3": 'Describe the image'
}

# The relations tied to each finetuning dataests
keyword_dict = {
    "LN_HF_Dataset.csv": [],
    "LeftRight_LN_HF_Dataset.csv": ["left", "right"],
    "LeftRight_v2_LN_HF_Dataset.csv": ["left", "right"],
    "Relations_LN_HF_Dataset.csv": ["below", "left", "right", "behind", "over"], 
    "Relations_v2_LN_HF_Dataset.csv": ["below", "left", "right", "behind", "over"]
}

# For loading pretrained models
best_pretrained_models = {
    "LN_HF_Dataset.csv": "fixed_memory/exp=fixed_memory:lora=16:data=LN_HF_Dataset.csv",
    "LeftRight_LN_HF_Dataset.csv": "fixed_memory/exp=fixed_memory:lora=16:data=LeftRight_LN_HF_Dataset.csv",
    "LeftRight_v2_LN_HF_Dataset.csv": "new_datasets/exp=new_datasets:lora=16:data=LeftRight_v2_LN_HF_Dataset.csv",
    "Relations_LN_HF_Dataset.csv": "fixed_memory/exp=fixed_memory:lora=16:data=Relations_LN_HF_Dataset.csv",
    "Relations_v2_LN_HF_Dataset.csv": "new_datasets/exp=new_datasets:lora=16:data=Relations_v2_LN_HF_Dataset.csv"
}

# For object extraction
skip_objects = ["left", "right", "side", "part", "parts", "lot", "background", "t", "sky", "top", "standing",
                "color", "picture",
                "center", "image", "air", "names", "view", "colors", "front", "mobile", "photo", "bottom",
                "middle", "colour",
                "foreground", "object", "objects", "steering", "name", "markings", "corner", "micro", "thing",
                "things", "nighttime", "G", "H", "W",
                "A", "B", "Option", "option"]

relation_list = ["below", "left", "right", "behind", "over"]
