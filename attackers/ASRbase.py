import torch
import torch.nn as nn
import pdb
import numpy as np
from utils import MAX_PER_DICT
from argparse import Namespace
from transformers import (
    Speech2Text2Tokenizer,
    Speech2Text2Processor,
    Speech2TextForConditionalGeneration,
)
from typing import Optional, List, Union
from DialogueAPI import dialogue

class BaseAttacker:
    def __init__(
        self,     
        device: Optional[torch.device] = None,
        tokenizer: Optional[Speech2Text2Tokenizer] = None,
        processor: Optional[Speech2Text2Processor] = None,
        model: Optional[Speech2TextForConditionalGeneration] = None,
        lr: float = 1e-3,
        max_len: int = 128,
        max_per: Optional[int] = None,
        max_iter: int = 5,
        att_norm: str = 'l2',
    ):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        if tokenizer is None:
            self.tokenizer = Speech2Text2Tokenizer.from_pretrained(
                "facebook/s2t-small-librispeech-asr"
            )
        else:
            self.tokenizer = tokenizer
        if processor is None:
            self.processor = Speech2Text2Processor.from_pretrained(
                "facebook/s2t-small-librispeech-asr"
            )
        else:
            self.processor = processor
        if model is None:
            self.model = Speech2TextForConditionalGeneration.from_pretrained(
                "facebook/s2t-small-librispeech-asr"
            ).to(self.device)
        else:
            self.model = model.to(self.device)
            
        self.bos_token_id = self.model.config.bos_token_id
        self.eos_token_id = self.model.config.eos_token_id
        self.pad_token_id = self.model.config.pad_token_id
        self.lr = lr
        self.max_len = max_len
        
        if max_per is None:
            self.max_per = MAX_PER_DICT[att_norm]
        else:
            self.max_per = max_per
            
        self.max_iter = max_iter
        self.att_norm = att_norm
        self.num_beams = self.model.config.num_beams
        self.num_beam_groups = self.model.config.num_beam_groups
        
        self.softmax = nn.Softmax(dim=1)
        self.flatten = nn.Flatten()
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.relu = nn.ReLU()

    @classmethod
    def _get_hparam(cls, namespace: Namespace, key: str, default=None):
        if hasattr(namespace, key):
            return getattr(namespace, key)
        print('Using default argument for "{}"'.format(key))
        return default

    def run_attack(self, audios: List[np.ndarray], sample_rate: int):
        pass

    def compute_loss(self, audios: List[np.ndarray], sample_rate: int):
        pass
    
    def compute_seq_len(self, seq: torch.Tensor):
        if seq.shape[0] == 0: # empty sequence
            return 0
        if seq[0].eq(self.pad_token_id):
            return int(len(seq) - sum(seq.eq(self.pad_token_id)))
        else:
            return int(len(seq) - sum(seq.eq(self.pad_token_id))) - 1
    
    def remove_pad(self, s: torch.Tensor):
        return s[torch.nonzero(s != self.pad_token_id)].squeeze(1)

    def process(
            self, 
            audios: Union[List[np.ndarray], np.ndarray], 
            sample_rate: int
        ):
        inputs = self.processor(
            audios,
            sampling_rate=sample_rate, 
            return_tensors="pt",
            padding=True,
        ) 
        # input_features, attention_mask
        input_features = inputs.input_features # B x T x F
        input_features = input_features.to(self.device)
        return input_features

    def inference(self, input_features: torch.Tensor): 
        outputs = dialogue(
            self.model,
            inputs=input_features,
            early_stopping=False,
            max_length=self.max_len,
            use_cache=True,
            num_beams=self.num_beams,
            num_beam_groups=self.num_beam_groups,
        )
        # sequences, sequences_scores, scores, beam_indices
        seqs = outputs['sequences'].detach()
        # print("seqs (before remove pad) ({}): {}".format([seq.shape for seq in seqs], seqs))
        seqs = [self.remove_pad(seq) for seq in seqs]
        # print("seqs (after remove pad) ({}): {}".format([seq.shape for seq in seqs], seqs))
        out_scores = outputs['scores']
        # print("seqs scores: {}".format(outputs['sequences_scores']))
        pred_len = [self.compute_seq_len(seq) for seq in seqs]
        # print("pred_len: {}".format(pred_len))
        return pred_len, seqs, out_scores

    def get_predictions(self, audios: List[np.ndarray], sample_rate: int):
        input_features = self.process(audios, sample_rate)
        pred_len, seqs, out_scores = self.inference(input_features)
        return pred_len, seqs, out_scores
    
    def get_ASR_string_len(
            self, 
            audios: Union[List[np.ndarray], torch.Tensor], 
            sample_rate: int = 16000,
        ):
        if isinstance(audios, torch.Tensor):
            pred_len, seqs, _ = self.inference(audios)
        else:
            pred_len, seqs, _ = self.get_predictions(audios, sample_rate)
        return seqs[0], pred_len[0]

    def get_ASR_len(
            self, 
            audios: Union[List[np.ndarray], torch.Tensor], 
            sample_rate: int = 16000,
        ):
        if isinstance(audios, torch.Tensor):
            pred_len, _, _ = self.inference(audios)
        else:
            pred_len, _, _ = self.get_predictions(audios, sample_rate)
        return pred_len

    def get_ASR_strings(
            self, 
            audios: Union[List[np.ndarray], torch.Tensor], 
            sample_rate: int = 16000,
        ):
        if isinstance(audios, torch.Tensor):
            pred_len, seqs, _ = self.inference(audios)
        else:
            pred_len, seqs, _ = self.get_predictions(audios, sample_rate)
        out_res = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in seqs]
        return out_res, pred_len
    
    
    

