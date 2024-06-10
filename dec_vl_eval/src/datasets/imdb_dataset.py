from datasets import load_dataset
import numpy as np
import spacy
from SVLC_learning.negs_and_pos import RandBothNegatives


class IMDB:

    def __init__(self, config):
        self.dataset_name = 'imdb' 
        self.num_sentences = config.max_n_sentences
        self.dataset = self.load_imdb_dataset()
    
    def load_imdb_dataset(self):
        raw_dataset = load_dataset('imdb')['train']
        negative_generator = RandBothNegatives()

        # Build a new dataset by iterating thorugh all of the text items
        # and build a dataset by randomly taking chunks from the text and then 
        # Pick a sentence to make the focus!
        proc_dataset = []
        for item in raw_dataset: 
            text = item["text"]
            text = text.replace("<br />","") # Weird artifact, get rid of it.
            sentences = text.split(".")
            # We want to sample a contiguous block of some number of sentences, which 
            # is as simple as sampling the number of sentences and the starting point.
            lower_bound = 3
            upper_bound = min(len(sentences), self.num_sentences)

            if upper_bound <= lower_bound:
                num_sentences = lower_bound 
            else:
                num_sentences = np.random.randint(lower_bound, upper_bound)

            if len(sentences) >= lower_bound:

                if len(sentences) == num_sentences:
                    starting_idx = 0
                else:
                    starting_idx = np.random.randint(0, len(sentences) - num_sentences)

                # Extract the block of text that we want to look at.
                new_text = sentences[starting_idx:starting_idx+num_sentences] 

                # Choose the sentence we want to tamper with.
                target_sentence_idx = np.random.randint(len(new_text))
                target_sentence = new_text[target_sentence_idx]

                # Now we will corrupt this sentence by inverting words.
                corrupt_sentence = negative_generator.create_negs(target_sentence) 

                # Sometimes the sentence just gets different spacing than the original,
                # so as a filter get rid of the spacing of both and check if they 
                # are the same.
                def standardize(sentence):
                    lowercase_sentence = sentence.lower()
                    without_spaces = lowercase_sentence.replace(" ", "")
                    return without_spaces

                def prune_output(sentence):
                    sent = sentence.replace("<start_of_text>", "")
                    sent = sent.replace("<end_of_text>", "")
                    return sent[0].upper() + sent[1:]

                if len(corrupt_sentence) > 0:
                    corrupt_sentence = prune_output(corrupt_sentence)
                    
                    if standardize(target_sentence) != standardize(corrupt_sentence):
                        # Now we flip a coint to determine if the target will be A or B.
                        print("Target:", target_sentence)
                        print("Corrupt:", corrupt_sentence)
                        print()

                        target_choice = 'A' if np.random.uniform() > 0.5 else 'B'
                        corrupt_choice = 'B' if target_sentence_idx == 'A' else 'A'

                        # Convert our sentence list back to a string
                        text = ".".join(new_text)

                        new_item = {
                            "full_caption": text,
                            "true_caption": target_sentence,
                            "true_label": target_choice,
                            "false_caption": corrupt_sentence,
                            "false_label": corrupt_choice
                        }
                        proc_dataset.append(new_item)

        return proc_dataset
    
    def negate_sentence(self, sentence):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(sentence)
        
        negated_sentence = []
        for token in doc:
            if token.pos_ == 'ADJ':
                negated_token = 'not ' + token.text
            else:
                negated_token = token.text
            
            negated_sentence.append(negated_token)
        
        return ' '.join(negated_sentence)
