import torch
from transformers.utils import PaddingStrategy
from dataclasses import dataclass
from typing import Optional, Union, Any
import argparse

from transformers import InstructBlipProcessor
from PIL import Image

_DEFAULT_EXCLUDE_ = ['target', 'pbp_file_path', 'url', '_stf', '_eff', '_pos', 'id', 'excitement', 'file_name', 'day_msg', '_nat', 'event_type', '_fn']

@dataclass
class InstructBlipCollator:
    args: argparse.Namespace
    vis_processor: InstructBlipProcessor
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = 512
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors

        image_inputs = []
        outputs = [f['output_text'] for f in features]
        for i, _ in enumerate(features):
            img = Image.open(features[i]['image_path']).convert("RGB")
            inputs = self.vis_processor(
                images=img,
                text=features[i]['input_text'],
                return_tensors="pt"
            )
            image_inputs.append(inputs.pop('pixel_values'))
            features[i] = inputs

        input_text_features = {k: [f[k][0].tolist() for f in features] for k in ['input_ids', 'attention_mask']}
        image_features = {k.replace('qformer_', ''): [f[k][0].tolist() for f in features] for k in ['qformer_input_ids', 'qformer_attention_mask']}

        labels = self.vis_processor.tokenizer(
            outputs,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        padded_text_feats = self.vis_processor.tokenizer.pad(
            input_text_features,
            padding='max_length',
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        padded_image_feats = self.vis_processor.tokenizer.pad(
            image_features,
            padding='max_length',
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        renamed_image_feats = {'qformer_'+k: padded_image_feats[k] for k in padded_image_feats}
        features_ = {**padded_text_feats, **renamed_image_feats}
        features_['pixel_values'] = torch.concatenate(image_inputs, dim=0)

        features_['labels'] = labels['input_ids']
        features_["labels"][features_["labels"] == self.vis_processor.tokenizer.pad_token_id] = -100

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features_["labels"])
            features_["decoder_input_ids"] = decoder_input_ids

        return features_
