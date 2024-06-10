# Rethinking the Evaluation of Compositional Reasoning for Modern VLMs

## 0. Environment Setup.

To setup the conda environment, use the following sequence of commands.

First, create a new environment.
```
conda create --name <name of environment>
```
Next, source into the environment to activate it..
```
. activate <name of environment>
```
Finally, install the necessary pip dependencies.
```
pip install -r requirements.txt
```

If needed, also run `export TRANSFORMERS_CACHE=<directory to server>/.cache` if you need to locate the huggingface cache to the server for access to compute resources.

## 1. Evaluation Script

To evaluate the different vision-language models on the original datasets, we can use the `eval.py` script. Since 
our work focuses on the instructblip-flan-t5, instructblip-vicuna-7b, and llava-v1.5-7b models, this readme file will 
detail the commands and steps for these models.

Example command to evaluate an instructblip model on the original SugarCrepe benchmark, using the perplexity inference evaluation method:
```
python eval.py --model <instructblip model endpoint> --eval_dataset SUGAR --select_by loss_score
```

Example command for llava-1.5 model:
```
python eval.py --model llava-v1.5-7b --eval_dataset SUGAR --select_by loss_score --temperature 0 --max_new_tokens 128 --prompt 11 --eval_base_dir <parent directory path>/sugar_results/ --experiment_name baseline --job_name sugar_llava-v1.5-7b
```

The results are saved in pickle file format, and the short script `dec_vl_eval/src/utils/pick_to_csv.py` can be used to unpack this into a csv as needed.

Here is an explanation of some different options for evaluation scenarios using different command line arguments. The `args.py`
file also contains brief descriptions for all available arg options.

### Model

```
--model (default='instructblip_flan_t5') 
--backend (default='hf')
```
This option determines the model which is evaluated and what 
backend to use. For our work, we focus on using huggingface instructblip models, where the endpoints are:
```
- instructblip_flan_t5
- instructblip_vicuna_7b 
```
as defined in `dec_vl_eval/src/models/vqa_models.py`.


### Evaluation dataset

Our code gives us the option to test a variety of different
VQA datasets, and in our work we focus on 3 partitions from SugarCrepe. To evaluate on a 
particular partition of a dataset, we note the following two arguments:

```
--eval_dataset (default='SUGAR')
--dataset_partition (default=None)
```

A number of datasets are defined in `dec_vl_eval/src/datasets/benchmark_datasets.py`, but here are the abbreviations for the SugarCrepes:

```
- SUGAR (Corresponding to SugarCrepe) (https://github.com/RAIVNLab/sugar-crepe)
    --eval_dataset SUGAR --dataset_partition replace-rel
    --eval_dataset SUGAR --dataset_partition replace-att
    --eval_dataset SUGAR --dataset_partition replace-obj
```

Finally, during the evaluation loop, we save a dataframe of 
statistics regarding the different parts of evaluation. 
We can control how often this is saved with the `save_every`
parameter and how many samples to take in total with `num_samples`.

```
--save_every (default=100)
--num_samples (default=10,000)
```

Please adapt the code to evaluate our ConMe dataset. 

### Scoring options

For the purposes of our work, we focus on the perplexity-based loss
score computation for our evaluation, method which is specified by

```
--select_by loss_score
```

`loss_score` scores the provided caption options by their negative
perplexity of the model (for additional details please refer to 
`dec_vl_eval/src/models/score.py`)

We complement the ConMe curation pipeline -- our automatic framework for mining hard CR QA, by also contributing an LLM-based analysis tool for automatic categorization of VLM mistakes according to human-specified error taxonomies provided simply by natural language description. This analysis pipeline is necessary, as our ConMe curation pipeline is automatic, adaptive (to the target VLMs being analyzed), and scalable (in a sense we can scale arbitrarily by providing the ConMe curation framework with more images). Hence, the generated CR QA in ConMe need to be analyzed (categorized) automatically in order to dynamically mine insights on the evaluated VLM weaknesses in terms of the relative distribution of their errors across error categories or other CR QA insights specified by the taxonomies. The code about this is contained in the `error_analysis.ipynb` jupyter notebook.

## 2. Generation of Hard Negatives

### Generating and verifying negatives from the LLM

We use the following steps and commands to generate a number of hard negatives per sample in each original dataset partition:

First, run the script for generating negatives (we use an API to access the Falcon-180B model). Here is an example command 
to generate negatives for the replace-rel partition of SugarCrepe:
```
python prompt_negatives.py --eval_dataset <dataset name> --partition <partition used to select prompt> --genai_key <API key>
```
The `prompt_negative.py` file also contains the contraction check verification for each sample, in addition to the prompt templates
for each type of dataset partition. 
Each sample has a generated csv file with its 5 LLM-generated negatives, so after this script finishes, we use python's pandas package to gather these into one csv file.

### VLM based hard negative selection

#### Loss Scores
Next, we compute the loss scores between each pairwise combination of original positive vs. candidate generated negative
using the following command templates:

For instructblip models:
```
python -m prompt_negatives.select_negatives --model <model endpoint> --eval_dataset SAMPLE_NEG --select_by loss_score --dataset_partition <dataset_partition>
```

For llava-1.5:
```
python -m prompt_negatives.select_negatives --model llava-v1.5-7b --eval_dataset SAMPLE_NEG --select_by loss_score --temperature 0 --max_new_tokens 128 --prompt 11 --dataset_partition <dataset_partition>
```

#### Selection of top 1 challenging negative
Thereafter, we can select and evaluate on the top k negatives for each model by using the following commands (default is top 1):

For instructblip models:
```
python  -m prompt_negatives.eval_negatives --model <model endpoint> --eval_dataset LLM --select_by loss_score --eval_base_dir <parent directory path>/prompt_negatives/sugar/ --experiment_name results --job_name <dataset_partition>_<model endpoint> --dataset_partition <dataset_partition>
```

For llava-1.5:
```
python  -m prompt_negatives.eval_negatives --model llava-v1.5-7b  --eval_dataset LLM --select_by loss_score --temperature 0 --max_new_tokens 128 --prompt 11 --eval_base_dir <parent directory path>/prompt_negatives/sugar/ --experiment_name results --job_name <dataset_partition>_llava-v1.5-7b --dataset_partition <dataset_partition> 
```
These now result in a new model-specific dataset for each original partition.

### Combined new dataset
After getting each model-specific dataset, we use the following command templates to obtain the combined datasets per 
partition (i.e. 3 SugarCrepe partitions).
The first command is used to join the selected negative with its specific contradictions as identified during the prompting process,
while the second command then concatenates the model-specific datasets together. For the leave-on-out datasets, we also 
utilize the `prompt_negatives/combined_datasets.py` script but modify it slightly to only include the model-specific datasets not corresponding to itself.
```
python -m prompt_negatives.combine_results_contradictions --model <model endpoint> --dataset_dir <parent name of dataset i.e. sugar> --dataset_partition <partition>
python -m prompt_negatives.combine_datasets
```

Lastly, to evaluate on the final resulting datasets, we use the following commands:

For instructblip models:
```
python eval.py --model <model endpoint> --eval_dataset COMBINED_LLM --select_by loss_score --eval_base_dir <output parent directory path> --experiment_name <dataset> --job_name combined_<dataset partition>_<model endpoint> --dataset_partition <dataset partition>
```

For llava-1.5 models:
```
python eval.py --model llava-v1.5-7b --eval_dataset COMBINED_LLM --select_by loss_score --temperature 0 --max_new_tokens 128 --prompt 11 --eval_base_dir <output parent directory path> --experiment_name <dataset> --job_name combined_<dataset partition>_<model endpoint> --dataset_partition <dataset partition>
```

## ConMe Dataset 

Our curated CR benchmark is hosted on this [HuggingFace Dataset Card](https://huggingface.co/conme/ConMe).

Which can simply be downloaded by running the following command:

```
git clone https://huggingface.co/conme/ConMe
```

## 3. LLM-Identification of Partition Labels

For our work on using LLMs to generate new partition labels for our datasets, we run a script to obtain the output from the contradiction check, which is then used as an input for the classification labeling.
Here is the prompt template, where we can specify either `--choice orig` or `--choice new` for which merged datasets to label, and 
we can optionally restrict the number of samples.
```
python -m prompt_negatives.classify_contradictions --choice orig --dataset_partition <partition> --genai_key <API key> --job_name <time.strftime("%Y%m%d_%H%M")> (--num_samples 100)
```

This also saves one output csv file per sample, so we can finally concatenate the results using 
```
python -m prompt_negatives.combine_labels --dataset_partition <partition>
```
### To cite us: 
```bibtex

@article{mirza2024mpvr,
    author    = {Huang, Irene* and Lin, Wei* and Mirza, M. Jehanzeb* and  Hansen, Jacob A. and Doveh, Sivan and Butoi, Victor Ion and Herzig, Roei and Arbelle, Assaf and Kuhene, Hilde and Darrel, Trevor and Gan, Chuang and Olivia, Aude and Feris, Rogerio and Karlinsky, Leonid},
    journal   = {ArXiv},
    title     = {{ConMe: Rethinking Evaluation of Compositional Reasoning for Modern VLMs}},
    year      = {2024}
    }
```
\* Equal Contribution.