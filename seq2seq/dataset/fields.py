import torch
import datasets
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from torch.utils.data import Dataset
from transformers.utils import PaddingStrategy
from transformers import AutoTokenizer, PreTrainedTokenizerBase



class Seq2SeqDataset(Dataset):
    def __init__(
        self, 
        data_file: str, 
        tokenizer: AutoTokenizer, 
        max_len: int = 128,
        src_field_name: str = 'src',
        tgt_field_name: str = 'tgt',
        
    ):
        self.max_len = max_len
        self.tokenizer = tokenizer
        df = pd.read_csv(data_file, sep='\t')
        self.dataset = datasets.Dataset.from_pandas(df)
        self.src_field_name = src_field_name
        self.tgt_field_name = tgt_field_name
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        instance = self.dataset[idx]
        src = instance[self.src_field_name]
        tgt = instance[self.tgt_field_name]
        src = self.tokenizer(
            src, 
            max_length=self.max_len, 
            truncation=True, 
            return_tensors='pt'
        )['input_ids'][0]
        tgt = self.tokenizer(
            tgt, 
            max_length=self.max_len, 
            truncation=True, 
            return_tensors='pt'
        )['input_ids'][0]
        return {
            'input_ids': src, 
            'input_lens': src.size(0),
            'labels': tgt,
            'label_lens': tgt.size(0),
        }



class Seq2SeqCollator:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index) among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        padding: Union[bool, str, PaddingStrategy] = True, 
        max_length: Optional[int] = None, 
        pad_to_multiple_of: Optional[int] = None, 
        label_pad_token_id: int = -100,
        return_tensors: str = "pt",
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
            
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        return features