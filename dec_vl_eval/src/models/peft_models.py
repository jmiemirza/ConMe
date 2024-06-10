from peft.tuners.multitask_prefix_tuning import MultitaskPrefixTuningConfig
from peft.utils import PeftType, TaskType
from transformers.trainer_utils import get_last_checkpoint
import torch
import os

def getPeftConfig(model_name: str, nVirtToks: int=100, encoder_prompting: bool=False):
    if model_name == 'google/flan-t5-large':
        # #https://huggingface.co/google/flan-t5-large/blob/main/config.json (when HF is installed editable under lib, we can probably find those files according to model name)
        peft_config = MultitaskPrefixTuningConfig(
            encoder_prompting=encoder_prompting,
            peft_type=PeftType.MULTITASK_PREFIX_TUNING,
            task_type=TaskType.SEQ_2_SEQ_LM,
            num_virtual_tokens=nVirtToks,
            token_dim=1024,
            num_transformer_submodules=2,
            num_attention_heads=16,
            num_layers=24,
            encoder_hidden_size=1024)
    elif model_name == 'google/flan-t5-base':
        # https://huggingface.co/google/flan-t5-base/blob/main/config.json (when HF is installed editable under lib, we can probably find those files according to model name)
        peft_config = MultitaskPrefixTuningConfig(
            encoder_prompting=encoder_prompting,
            peft_type=PeftType.MULTITASK_PREFIX_TUNING,
            task_type=TaskType.SEQ_2_SEQ_LM,
            num_virtual_tokens=nVirtToks,
            token_dim=768,
            num_transformer_submodules=2,
            num_attention_heads=12,
            num_layers=12,
            encoder_hidden_size=768)
    elif model_name == 'google/flan-t5-xl':
        # https://huggingface.co/google/flan-t5-xl/blob/main/config.json (when HF is installed editable under lib, we can probably find those files according to model name)
        peft_config = MultitaskPrefixTuningConfig(
            encoder_prompting=encoder_prompting,
            peft_type=PeftType.MULTITASK_PREFIX_TUNING,
            task_type=TaskType.SEQ_2_SEQ_LM,
            num_virtual_tokens=nVirtToks,
            token_dim=2048,
            num_transformer_submodules=2,
            num_attention_heads=32,
            num_layers=24,
            encoder_hidden_size=2048)
    elif model_name == 'facebook/bart-large':
        # https://huggingface.co/facebook/bart-large/blob/main/config.json (when HF is installed editable under lib, we can probably find those files according to model name)
        peft_config = MultitaskPrefixTuningConfig(
            peft_type=PeftType.MULTITASK_PREFIX_TUNING,
            task_type=TaskType.SEQ_2_SEQ_LM,
            num_virtual_tokens=nVirtToks,
            token_dim=1024,
            num_transformer_submodules=2,
            num_attention_heads=16,
            num_layers=12,
            encoder_hidden_size=1024)
    elif model_name == 'facebook/bart-base':
        # https://huggingface.co/facebook/bart-large/blob/main/config.json (when HF is installed editable under lib, we can probably find those files according to model name)
        peft_config = MultitaskPrefixTuningConfig(
            peft_type=PeftType.MULTITASK_PREFIX_TUNING,
            task_type=TaskType.SEQ_2_SEQ_LM,
            num_virtual_tokens=nVirtToks,
            token_dim=768,
            num_transformer_submodules=2,
            num_attention_heads=12,
            num_layers=6,
            encoder_hidden_size=3072)
    else:
        raise ValueError(f'Unsupported model: {model_name}')
    return peft_config

def multi_lora_load_from_checkpoints(resume_from_checkpoints, model):
    for index, path in enumerate(resume_from_checkpoints):
        if isinstance(path, list):
            identifier = path[0]
            path = path[1]
        else:
            identifier = None
        print(f'*** Loading lora PEFT delta from {path}, identifier: {identifier}')
        last_chkpt_path = get_last_checkpoint(path)
        state_dict = torch.load(os.path.join(last_chkpt_path, 'adapter_model.bin'), map_location="cpu")
        model.load_state_dict(state_dict, False)
        model.add_lora_modules(identifier=identifier)
    return model