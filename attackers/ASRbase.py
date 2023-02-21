import torch
import torch.nn as nn
import pdb
import numpy as np
from utils import MAX_PER_DICT
from argparse import Namespace
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.nnet.containers import LengthsCapableSequential
from speechbrain.decoders import S2SBeamSearcher
from speechbrain.lobes.models.transformer.TransformerLM import TransformerLM


class BaseAttacker:
    def __init__(
        self,     
        device: torch.device = None,
        model: EncoderDecoderASR = None,
        lr: float = 1e-3,
        max_len: int = 128,
        max_per: int = None,
        max_iter: int = 5,
        att_norm: str = 'l2',
    ):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        if model is None:
            self.model = EncoderDecoderASR.from_hparams("speechbrain/asr-transformer-transformerlm-librispeech")
        else:
            self.model = model
        self.encoder: LengthsCapableSequential = model.mods.encoder
        self.decoder: S2SBeamSearcher = model.mods.decoder
        self.lm_model: TransformerLM = model.mods.lm_model
        self.eos_token_id = self.decoder.eos_index
        self.pad_token_id = self.decoder.eos_index
        
        self.lr = lr
        self.max_len = max_len
        if max_per is None:
            self.max_per = MAX_PER_DICT[att_norm]
        else:
            self.max_per = max_per
        self.max_iter = max_iter
        self.att_norm = att_norm
        self.beam_size = self.decoder.beam_size
        
        self.softmax = nn.Softmax(dim=1)
        self.flatten = nn.Flatten()
        self.bce_loss = nn.BCELoss(reduction='none')
        self.mse_Loss = nn.MSELoss(reduction='none')
        self.relu = nn.ReLU()

    @classmethod
    def _get_hparam(cls, namespace: Namespace, key: str, default=None):
        if hasattr(namespace, key):
            return getattr(namespace, key)
        print('Using default argument for "{}"'.format(key))
        return default

    def run_attack(self, audios: torch.Tensor, wav_lens: torch.Tensor):
        pass

    def compute_loss(self, audios: torch.Tensor, wav_lens: torch.Tensor):
        pass

    def get_ASR_len(self, audios: torch.Tensor, wav_lens: torch.Tensor):
        pred_tokens, _, _ = self.get_predictions(audios, wav_lens)
        pred_lens = np.array([len(pred) for pred in pred_tokens])
        return pred_lens

    def get_predictions(self, audios: torch.Tensor, wav_lens: torch.Tensor):
        encoder_out = self.model.encode_batch(audios, wav_lens) # B X T X D
        pred_tokens, topk_scores, scores = self.model.mods.decoder(encoder_out, wav_lens)
        return pred_tokens, topk_scores, scores

