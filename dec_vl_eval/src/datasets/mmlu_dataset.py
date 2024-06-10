import os
import pandas as pd


class MMLU:

    def __init__(self, root_dir, n_shot=5):

        self.TASKS = [
            'abstract_algebra',
            'anatomy',
            'astronomy',
            'business_ethics',
            'clinical_knowledge',
            'college_biology',
            'college_chemistry',
            'college_computer_science',
            'college_mathematics',
            'college_medicine',
            'college_physics',
            'computer_security',
            'conceptual_physics',
            'econometrics',
            'electrical_engineering',
            'elementary_mathematics',
            'formal_logic',
            'global_facts',
            'high_school_biology',
            'high_school_chemistry',
            'high_school_computer_science',
            'high_school_european_history',
            'high_school_geography',
            'high_school_government_and_politics',
            'high_school_macroeconomics',
            'high_school_mathematics',
            'high_school_microeconomics',
            'high_school_physics',
            'high_school_psychology',
            'high_school_statistics',
            'high_school_us_history',
            'high_school_world_history',
            'human_aging',
            'human_sexuality',
            'international_law',
            'jurisprudence',
            'logical_fallacies',
            'machine_learning',
            'management',
            'marketing',
            'medical_genetics',
            'miscellaneous',
            'moral_disputes',
            'moral_scenarios',
            'nutrition',
            'philosophy',
            'prehistory',
            'professional_accounting',
            'professional_law',
            'professional_medicine',
            'professional_psychology',
            'public_relations',
            'security_studies', 
            'sociology',
            'us_foreign_policy',
            'virology',
            'world_religions'
        ]

        self.choices = ["A", "B", "C", "D"]
        self.data_root = root_dir 
        self.num_train = int(n_shot)
        self.dataset = self.load_dataset()


    def format_subject(self, subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s

    def format_example(self, df, idx, include_answer=True):
        prompt = df.iloc[idx, 0]
        k = df.shape[1] - 2
        for j in range(k):
            prompt += "\n{}. {}".format(self.choices[j], df.iloc[idx, j+1])

        if include_answer:
            prompt += "\nAnswer: {}\n\n".format(df.iloc[idx, k + 1])

        return prompt

    def gen_prompt(self, train_df, subject, k=-1):
        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(self.format_subject(subject))

        if k == -1:
            k = train_df.shape[0]

        for i in range(k):
            prompt += self.format_example(train_df, i)

        return prompt

    def batch_split(self, prompts, batch_num):
        batch_prompts = []
        mini_batch = []
        for prompt in prompts:
            mini_batch.append(prompt)
            if len(mini_batch) == batch_num:
                batch_prompts.append(mini_batch)
                mini_batch = []
        if len(mini_batch) != 0:
            batch_prompts.append(mini_batch)
        return batch_prompts

    def load_dataset(self):

        total_dataset = []

        for task in self.TASKS:

            print('Testing %s ...' % task)
            dev_df = pd.read_csv(os.path.join(self.data_root, "dev", task + "_dev.csv"), header=None)[:self.num_train]
            test_df = pd.read_csv(os.path.join(self.data_root, "test", task + "_test.csv"), header=None)

            for i in range(test_df.shape[0]):

                # get prompt and make sure it fits
                prompt_end = self.format_example(test_df, i, include_answer=False)
                train_prompt = self.gen_prompt(dev_df, task, self.num_train)

                prompt = train_prompt + prompt_end
                label = test_df.iloc[i, test_df.shape[1]-1]

                item_dict = {
                    "prompt": prompt,
                    "label": label,
                    "task": task 
                }
                total_dataset.append(item_dict)

        return total_dataset
