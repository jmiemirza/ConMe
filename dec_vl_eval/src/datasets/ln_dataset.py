import numpy as np
from tqdm import tqdm
import json
from SVLC_learning.negs_and_pos import RandBothNegatives


class GeneratorDataset:

    def __init__(self, data, min_sentences, max_neg_flips):
        self.caption_data = data
        self.caption_ids = list(data.keys())
        self.min_sentences = min_sentences
        self.max_negation_flips = max_neg_flips
        self.negative_generator = RandBothNegatives()
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        while self.index < len(self.caption_ids):
            item = self.caption_data[self.caption_ids[self.index]]
            try:
                processed_item = self.process_item(item)
                self.index += 1
                return processed_item
            except Exception:
                self.index += 1
                continue
        raise StopIteration

    def is_one_char_difference(self, string1, string2):
        if len(string1) != len(string2):
            return False

        diff_count = 0
        for char1, char2 in zip(string1, string2):
            if char1 != char2:
                diff_count += 1
                if diff_count > 1:
                    return False
        return True 

    def process_item(self, item):
        # Placeholder for the actual processing logic
        # You can replace this with your own implementation
        # or subclass the Dataset class and override this method
        # Some preprocessing to cleanup
        sentences = item.split(".")

        # We want to sample a contiguous block of some number of sentences, which is as simple as sampling the number of sentences and the starting point.
        upper_bound = min(len(sentences), self.min_sentences)
        
        num_sentences = self.min_sentences if (upper_bound == self.min_sentences) else np.random.randint(self.min_sentences, upper_bound)
        
        # Pick the starting sentence
        starting_idx = 0 if (len(sentences) == num_sentences) else np.random.randint(0, len(sentences) - num_sentences)
    
        # Extract the block of text that we want to look at.
        new_text = sentences[starting_idx:starting_idx+num_sentences] 
    
        # Choose the sentence we want to tamper with.
        target_sentence_idx = np.random.randint(len(new_text))
        target_sentence = new_text[target_sentence_idx]
    
        # Now we will corrupt this sentence by inverting words.
        num_flips = 1 if (self.max_negation_flips == 1) else np.random.randint(1, self.max_negation_flips)
    
        # Corrupt the sentence by negating various aspects of it.
        assert len(target_sentence) > 0
        corrupt_sentence = self.negative_generator.create_negs(target_sentence, num_neg_flips=num_flips) 
    
        if len(corrupt_sentence) > 0:
            corrupt_sentence = self.prune_output(corrupt_sentence)
            corrupt_sentence = corrupt_sentence.replace(" - ", "-")
            if not self.is_one_char_difference(self.standardize(target_sentence), self.standardize(corrupt_sentence)):
                target_choice = 'A' if np.random.uniform() > 0.5 else 'B'
                corrupt_choice = 'B' if target_choice == 'A' else 'A'

                # Convert our sentence list back to a string
                text = ".".join(new_text)
                new_item = {
                    "full_caption": text,
                    "true_caption": target_sentence,
                    "true_label": target_choice,
                    "false_caption": corrupt_sentence,
                    "false_label": corrupt_choice
                }

                return new_item
            else:
                raise Exception("Processing failed for item")
        else:
            raise Exception("Processing failed for item")

    def __len__(self):
        return len(self.caption_ids)


    # Sometimes the sentence just gets different spacing than the original,
    # so as a filter get rid of the spacing of both and check if they 
    # are the same.
    def standardize(self, sentence):
        lowercase_sentence = sentence.lower()
        without_spaces = lowercase_sentence.replace(" ", "")
        return without_spaces

    def prune_output(self, sentence):
        sent = sentence.replace(" i ", " I ")
        sent = sent.replace("<start_of_text>", "")
        sent = sent.replace(" <end_of_text>", "")
        return sent[0].upper() + sent[1:]


class LN_Captions:

    def __init__(self, config):
        self.dataset_name = 'localized-narratives captions' 
        self.num_sentences = config.max_n_sentences
        self.min_sentences = 3
        self.max_negation_flips = int(config.max_neg_flips)
        self.dataset_root = "/dccstor/leonidka1/victor_space/data/OpenImages"
        self.caption_dataset = self.load_caption_dataset()
        self.dataset = GeneratorDataset(
            self.caption_dataset,
            self.min_sentences,
            self.max_negation_flips
        )
        print("Loaded Localilized Captions Dataset with this many examples: ", len(self.caption_dataset))
    
    def load_caption_dataset(self):

        def load_jsonl(file_path):
            data = []
            with open(file_path, 'r') as file:
                for line in tqdm(file, desc="Loading File"):
                    record = json.loads(line)
                    data.append(record)
            return data

        json_path = self.dataset_root + "/open_images_train_v6_captions.jsonl"
        raw_dataset = load_jsonl(json_path)

        # Build a new dataset by iterating thorugh all of the text items
        # and build a dataset by randomly taking chunks from the text and then 
        # Pick a sentence to make the focus!
        image_captions = {}
        for item in tqdm(raw_dataset, desc='Extracting Captions'):
            image_id = item["image_id"] 
            caption = item["caption"]
            if image_id in image_captions.keys():
                image_captions[image_id] += caption
            else:
                image_captions[image_id] = caption

        # Filter the captions to only include those that have the minimum number
        # of sentences.
        lower_bound = 3
        all_keys = list(image_captions.keys())
        for image_id in tqdm(all_keys, desc='Filtering examples'):
            text = image_captions[image_id]
            sentences = text.split(".")

            # Eliminate broken sentences.
            sentences = [sen for sen in sentences if sen != ''] 

            if len(sentences) < lower_bound:
                image_captions.pop(image_id)
            else:
                new_image_text = ".".join(sentences)
                text = new_image_text.replace(", ", " ")
                text = text.replace(" - ", "-")
                image_captions[image_id] = text

        return image_captions
    
    def __len__(self):
        return len(self.caption_dataset)

