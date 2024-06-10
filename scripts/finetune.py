import importlib.util
from cvar_pyutils.debugging_tools import set_remote_debugger
cvar_pyutils_spec = importlib.util.find_spec("cvar_pyutils")
cvar_pyutils_found = cvar_pyutils_spec is not None

# local imports
from dec_vl_eval.src.datasets.collate import InstructBlipCollator
from args import get_args

# huggingface imports
from datasets import Dataset
import evaluate
from peft import get_peft_model
from peft import LoraConfig, TaskType
import transformers as trans
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import Blip2ForConditionalGeneration, Blip2Processor

# misc imports
import numpy as np
import os
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import time
import torch


_DEFAULT_EXCLUDE_ = ['target', 'pbp_file_path', 'url', '_stf', '_eff', '_pos', 'id', 'excitement', 'file_name', 'day_msg', '_nat', 'event_type', '_fn']

def is_float(element: any) -> bool:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False

def val_format(v):
    if isinstance(v, str):
        if (not v.isnumeric()) and is_float(v):
            v = f'{float(v):.2f}'
    return v

def is_excluded(x, exclude):
    for y in exclude:
        if y in x:
            return True
    return False

def dict2str(d, exclude=_DEFAULT_EXCLUDE_, no_shots=False):
    res = ''

    if no_shots:
        _ = d.pop('shots')

    for k in d:
        if not is_excluded(k, exclude):
            v = d[k]
            if v is not None:
                res += f'{k.replace("_", " ")}:{val_format(v) if not isinstance(v, dict) else dict2str(v, exclude)}[SEP]'
    return res if len(res) < 5 else res[:-5] # remove the last [SEP]

def prep_model_inputs(tokenizer, inputs, targets):
    prefix = ""  # TODO
    inputs = [prefix + inp for inp in inputs]
    model_inputs = tokenizer(
        inputs,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True
    )

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=targets, max_length=tokenizer.model_max_length, padding="max_length",
                       truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["output_text"]

    return model_inputs

def tokenize_function(examples, tokenizer):
    inputs = [dict2str(x) for x in examples]
    targets = [x['output_text'] for x in examples]

    return prep_model_inputs(tokenizer, inputs, targets)

def model_prep(args, device):

    if args.model == "instructblip_flan_t5":
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        vis_processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
    elif args.model == "instructblip_vicuna_7b":
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        vis_processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
    elif args.model == "instructblip_vicuna_13b":
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-13b")
        vis_processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-13b")
    elif args.model == "blip2_coco":
        model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b-coco')
        vis_processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b-coco')
    else:
        raise ValueError("Model not recognized.")

    model.to(device)

    return model, vis_processor


if __name__ == '__main__':
    os.environ['tokenizers_parallelism'] = '0'

    # args
    args = get_args()

    # setup debug
    if args.debug:
        set_remote_debugger(None, args.debug_port)

    # device
    device = torch.device(args.device)

    # seed
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    # tokenizer and model prep
    model, vis_processor = model_prep(args, device)
    print(f'loading the {args.dataset} dataset')
    data_root = "/dccstor/leonidka1/victor_space/data/LocalizedNarratives/"
    csv_path = data_root + args.finetune_dataset_csv
    dataset_df = pd.read_csv(csv_path)

    # Split dataset into train, val, and test sets
    train_val_ratio = 0.999
    train_dataset, val_dataset = train_test_split(
        dataset_df,
        test_size=(1 - train_val_ratio),
        shuffle=True
    )

    # Get it in the form we like
    train_dataset = Dataset.from_pandas(train_dataset)
    val_dataset = Dataset.from_pandas(val_dataset)

    print(f'finished loading the {args.dataset} dataset, splits:\nTrain: {len(train_dataset)} Val: {len(val_dataset)}')

    # peft
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=args.lora_target_modules
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Logging details
    if args.job_name is None:
        args.job_name = time.strftime("%Y%m%d_%H%M%S")

    checkpoint_dir = f"{args.train_base_dir}/checkpoint_logs/{args.experiment_name}/{args.job_name}"
    log_dir = f"{args.train_base_dir}/tensorboard_logs/{args.experiment_name}/{args.job_name}"

    # If the args.result_dir does not exist, create it.
    for dire in [checkpoint_dir, log_dir]:
        if not os.path.exists(dire):
            os.makedirs(dire)

    # Define the Trainer class
    TrainerClass = Seq2SeqTrainer
    ArgumentClass = Seq2SeqTrainingArguments
    trans.trainer.WEIGHTS_NAME = 'adapter_model.bin'

    # inference
    eval_cache_dir = os.path.join(checkpoint_dir, 'eval_cache')
    os.makedirs(eval_cache_dir, exist_ok=True)
    rouge = evaluate.load('rouge', cache_dir=eval_cache_dir)

    # Define the training arguments
    training_args = Seq2SeqTrainingArguments(
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        num_train_epochs=args.num_epochs,
        label_smoothing_factor=0.0,
        learning_rate=args.init_lr,
        per_device_train_batch_size=args.train_batch_sz,
        per_device_eval_batch_size=args.eval_batch_sz,
        save_total_limit=1,
        logging_steps=20,
        save_steps=100,
        output_dir=checkpoint_dir,
        logging_dir=log_dir,
        save_on_each_node=False,
        log_on_each_node=False,
        dataloader_drop_last=True,
        load_best_model_at_end=False,
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        local_rank=args.local_rank,
        fp16=False,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to='tensorboard'
    )

    def compute_metrics(eval_pred):
        # predicted_texts, gold_texts = eval_pred.predictions[0], eval_pred.label_ids
        if isinstance(eval_pred.predictions, tuple):
            predicted_texts, gold_texts = eval_pred.predictions[0], eval_pred.label_ids
        else:
            predicted_texts, gold_texts = eval_pred.predictions, eval_pred.label_ids

        predicted_texts = predicted_texts.argmax(axis=2)
        predicted_texts_, gold_texts_ = [], []
        for p, g in zip(predicted_texts, gold_texts):
            mask = (g >= 0)
            p = vis_processor.tokenizer.decode(p[mask])
            g = vis_processor.tokenizer.decode(g[mask])
            predicted_texts_.append(p)
            gold_texts_.append(g)
        print('### Val sample ###')
        for iS in range(5):
            print(f'Gold:{gold_texts_[iS]}\nPred:{predicted_texts_[iS]}')
        return rouge.compute(predictions=predicted_texts_, references=gold_texts_)

    collator = InstructBlipCollator(args, vis_processor)
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=collator
    )
    trainer.pft_args = args

    # Run the training model
    trainer.train()

