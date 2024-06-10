from peft  import PeftModel
from dec_vl_eval.src.utils.directory import keyword_dict, best_pretrained_models, skip_objects, relation_list

# misc imports
import numpy as np
import pathlib
import spacy


def get_caption_model(
    args,
    base_model,
    device
):
    model_dir = best_pretrained_models[args.caption_dataset]
    pretrained = pathlib.Path(f"{args.pretrained_root}/{model_dir}")
    checkpoint_dir = str(list(pretrained.glob("checkpoint*"))[0])
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model = model.to(device)
    return model


def model_add_captions(
    args,
    caption_model,
    sample,
    vis_processor,
    device
):
    if len(keyword_dict[args.caption_dataset]) > 0:
        contains_caption_keyword = np.any([kw in sample["prompt"] for kw in keyword_dict[args.caption_dataset]])
    else:
        contains_caption_keyword = True

    if contains_caption_keyword:
        caption_generate_settings = {
            "length_penalty": 1.0,
            "repetition_penalty": 1.0,
            "num_beams": 5,
            "max_length": 300,
            "min_length": 1,
            "top_p": 0.9,
            "temperature": 1.0
        }
        prompt = sample["prompt"]

        def extract_object_prompt(sentence):

            # Load the English model for spaCy
            nlp = spacy.load("en_core_web_sm")

            # Process the sentence
            doc = nlp(sentence)

            # Extract tangible objects (nouns) from the sentence
            objects = []
            for token in doc:
                if token.pos_ == "NOUN" and token.is_alpha and token.text not in skip_objects:
                    objects.append(token.text)
            unique_objs = list(np.unique(objects))

            if "v2" in args.caption_dataset.lower(): 
                # Extract tangible objects (nouns) from the sentence
                unique_rels = []
                for word in sentence.split(" "):
                    lower_word = word.lower()
                    if lower_word in relation_list and lower_word not in unique_rels:
                        unique_rels.append(lower_word)

                # Build the combined prompt
                obj_string = ", ".join(unique_objs)
                rel_string = ", ".join(unique_rels)
                cap_prompt = "Objects: " + obj_string + ". Relations: " + rel_string
            else:
                cap_prompt = ", ".join(unique_objs)

            return cap_prompt 

        caption_prompt = extract_object_prompt(prompt)
        caption_sample = {
            "image": sample["image"],
            "prompt": caption_prompt 
        }

        print("Caption Prompt:", caption_prompt)
        generated_text = model_generate(
            args,
            caption_model,
            caption_sample,
            vis_processor,
            caption_generate_settings,
            device
        )
        sample["prompt"] = generated_text + ". " + sample["prompt"]

    return sample
