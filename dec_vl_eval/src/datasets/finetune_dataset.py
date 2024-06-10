import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import spacy
import csv
import os
from dec_vl_eval.src.utils.directory import skip_objects, relation_list


def pretty_print_dict(dictionary):
    for key, value in dictionary.items():
        print(f"{key}: {value}")
    print()


class LocalizedObjects:

    def __init__(
        self,
        dataset_size,
        write_dataset,
        datapoint_type,
        csv_name
    ):
        self.dataset_name = 'localized-narratives captions' 
        self.dataset_root = "/dccstor/leonidka1/victor_space/data/LocalizedNarratives"
        self.caption_dataset = self.load_caption_dataset()
        self.data_list = []
        self.csv_name = csv_name
        self.write_dataset = write_dataset
        self.datapoint_type = datapoint_type
        if datapoint_type == "objects":
            self.datapoint_method = self.objects_target_method
        elif datapoint_type == "objs_and_rels":
            self.datapoint_method = self.object_and_relations_method
        else:
            raise NotImplementedError("Haven't implemented this way of building the dataset.")

        self.build_object_dataset(dataset_size)
    
    def load_caption_dataset(self):

        def load_jsonl(file_path):
            data = []
            with open(file_path, 'r') as file:
                for line in tqdm(file, desc="Loading File"):
                    record = json.loads(line)
                    data.append(record)
            return data

        json_path = self.dataset_root + "/full_coco_train_localized_narratives.jsonl"
        # json_path = self.dataset_root + "/coco_train_localized_narratives-00000-of-00004.jsonl"
        raw_dataset = load_jsonl(json_path)

        # Build a new dataset by iterating through the text items
        # and build a dataset by randomly taking chunks from the text and then 
        # Pick a sentence to make the focus!
        image_captions = {}
        for item in tqdm(raw_dataset, desc='Extracting Captions'):
            assert item['dataset_id'] == "mscoco_train2017"
            image_id = item["image_id"] 
            caption = item["caption"]
            if image_id in image_captions.keys():
                image_captions[image_id].append(caption)
            else:
                image_captions[image_id] = [caption]

        return image_captions

    def build_object_dataset(
        self,
        dataset_size
    ):

        image_root = "/dccstor/leonidka1/victor_space/data/coco/images/train2017/000000"
        csv_file_path = f"/dccstor/leonidka1/victor_space/data/LocalizedNarratives/{self.csv_name}"
        field_names = ["input_text", "output_text", "image_path"]

        if os.path.exists(csv_file_path):
            # Load the CSV file into a pandas DataFrame
            print("FOUND CSV, LOADING...")
            file_data = pd.read_csv(csv_file_path)
            saved_file_names = list(file_data['image_path'])
            self.data_list = file_data.to_dict(orient='records')
            print("LOADED CSV!")
        else:
            saved_file_names = []

        if dataset_size == -1:
            dataset_size = len(self.caption_dataset.keys())
        
        for ik_idx, image_key in tqdm(
            enumerate(self.caption_dataset.keys()),
            total=dataset_size,
            desc="Creating Noun Objects."
        ):

            file_name = image_root + image_key + ".jpg"
            captions = self.caption_dataset[image_key]

            if file_name not in saved_file_names:
                # Make sure you have enough sentences to sample from.
                if os.path.exists(file_name):
                    
                    # Choose the objects you want to get sentences from.
                    selected_caption = np.random.choice(captions)
                    sentences = selected_caption.split(".")
                    if len(sentences) == 1:
                        selected_sentence = selected_caption
                    else:
                        selected_sentence = np.random.choice(sentences[:-1])
        
                    # Extract the nouns from each of these sentences, we want to
                    # construct a dictionary that lets us go from noun -> [sentence with noun].
                    # If a noun is in multiple sentences, it will be appended to the list.
                    noun_dict ={
                        "image_path": file_name,
                        "selected_sentence": selected_sentence,
                    }

                    sen = str(selected_sentence)
                    sen_nouns = self.extract_objects(sen)
                    sen_rels = self.extract_relations(sen)

                    if len(sen_nouns) > 0:
                        # Keep track of the sentence indices in which these nouns appear.
                        unique_objects = list(
                            np.unique([sn.lower() for sn in sen_nouns if sn.lower() not in skip_objects])
                        )
                        unique_relations = list(np.unique(sen_rels))

                        if self.datapoint_type == "objects":
                            cond = (len(unique_objects) > 0)
                        elif self.datapoint_type == "objs_and_rels":
                            cond = (len(unique_objects) > 0) and (len(unique_relations) > 0)
                        else:
                            raise NotImplementedError("Haven't implemented this kind of dataset yet.")

                        if cond:
                            noun_dict["objects"] = unique_objects
                            noun_dict["relations"] = unique_relations

                            # Finally add this noun object to the dataset
                            new_obj = self.datapoint_method(noun_dict)

                            # Printing for testing
                            if not self.write_dataset:
                                pretty_print_dict(new_obj)

                            self.data_list.append(new_obj)
                            saved_file_names.append(new_obj['image_path'])
            
                            if len(self.data_list) > dataset_size:
                                # If you exceed the dataset size, break out.
                                break

                if ik_idx % 100 == 0 and self.write_dataset:
                    # Save intermediately so you don't have to wait until the end
                    self.write_to_csv(csv_file_path, field_names, self.data_list)
            else:
                print(f"Skipping {file_name}! (Already in CSV)")

        # Write one more time at the end
        if self.write_dataset:
            self.write_to_csv(csv_file_path, field_names, self.data_list)

    @staticmethod
    def objects_target_method(noun_dict):
        obj_string = ", ".join(noun_dict["objects"])
        new_obj = {
            "input_text": obj_string,
            "output_text": noun_dict["selected_sentence"],
            "image_path": noun_dict["image_path"]
        }
        return new_obj

    @staticmethod
    def object_and_relations_method(noun_dict):
        obj_string = ", ".join(noun_dict["objects"])
        rel_string = ", ".join(noun_dict["relations"])
        new_obj = {
            "input_text": "Objects: " + obj_string + ". Relations: " + rel_string,
            "output_text": noun_dict["selected_sentence"],
            "image_path": noun_dict["image_path"]
        }
        return new_obj

    @staticmethod
    def extract_objects(sentence):
        # Load the English model for spaCy
        nlp = spacy.load("en_core_web_sm")
        
        # Process the sentence
        doc = nlp(sentence)
        
        # Extract tangible objects (nouns) from the sentence
        objects = []
        for token in doc:
            if token.pos_ == "NOUN" and token.is_alpha:
                objects.append(token.text)
        
        return objects

    # kind of stupid way to do this, just manually extracting some of the relations which we are about.
    def extract_relations(self, sentence):
        # Extract tangible objects (nouns) from the sentence
        relations = []
        for word in sentence.split(" "):
            if word.lower() in relation_list:
                relations.append(word.lower())

        return relations

    @staticmethod
    def write_to_csv(csv_file_path, field_names, data_list):
        with open(csv_file_path, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
        
            # Write the header row (field names)
            writer.writeheader()
        
            # Write the data rows
            for data in data_list:
                writer.writerow(data)
    
    def __len__(self):
        return len(self.caption_dataset)
