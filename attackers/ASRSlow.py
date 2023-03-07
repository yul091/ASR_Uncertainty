import sys
sys.path.append("..") # Adds higher directory to python modules path.
import numpy as np
from tqdm import tqdm
from math import ceil
import logging
from argparse import Namespace
import torch
import torch.nn.functional as F
import librosa
from torch.optim import Adam
from torchaudio.utils import download_asset
from typing import Optional, List, Union
from .ASRbase import BaseAttacker
from uncertainty import (
    data_uncertainty,
    activate_mc_dropout,
    convert_dropouts,
    bald,
    probability_variance,
    sampled_max_prob,
)
from utils import SAMPLE_NOISE
from transformers import (
    Speech2Text2Tokenizer,
    Speech2Text2Processor,
    Speech2TextForConditionalGeneration,
)
logger = logging.getLogger(__name__)


class ASRSlowAttacker(BaseAttacker):
    
    def __init__(
        self, 
        device: torch.device = None,
        tokenizer: Speech2Text2Tokenizer = None,
        processor: Speech2Text2Processor = None,
        model: Speech2TextForConditionalGeneration = None,
        lr: float = 1e-3,
        max_len: int = 128,
        max_per: int = None,
        max_iter: int = 5,
        att_norm: str = 'l2',
        noise_file: str = None,
        committee_size: int = 10,
        data_uncertainty: str = 'vanilla',
        model_uncertainty: str = 'prob_variance',
    ):
        super(ASRSlowAttacker, self).__init__(
            device=device, 
            tokenizer=tokenizer, 
            processor=processor, 
            model=model, 
            max_len=max_len, 
            lr=lr, 
            max_per=max_per, 
            max_iter=max_iter, 
            att_norm=att_norm,
        )
        if noise_file is not None:
            self.noise = download_asset(noise_file)
        else:
            self.noise = download_asset(SAMPLE_NOISE)

        self.model_uncertainty = model_uncertainty
        self.data_uncertainty = data_uncertainty
        self.committee_size = committee_size
        dropout_args = Namespace(
            max_n=100,
            max_frac=0.4,
            mask_name="mc",
            dry_run_dataset="train",
        )
        self.uncertainty_args = Namespace(
            dropout_type="MC",
            data_ue_type=self.data_uncertainty,
            inference_prob=0.1,
            committee_size=self.committee_size,  # number of forward passes
            dropout_subs="last",
            eval_bs=1000,
            use_cache=True,
            eval_passes=False,
            dropout=dropout_args,
        )
    
    def add_noise(
        self,
        waveform: torch.Tensor, 
        noise: torch.Tensor, 
        snr: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None,
    ):
        if not (waveform.ndim - 1 == noise.ndim - 1 == snr.ndim and (lengths is None or lengths.ndim == snr.ndim)):
            raise ValueError("Input leading dimensions don't match.")

        L = waveform.size(-1)
        if L != noise.size(-1):
            raise ValueError(f"Length dimensions of waveform and noise don't match (got {L} and {noise.size(-1)}).")

        # compute scale
        if lengths is not None:
            mask = torch.arange(0, L, device=lengths.device).expand(waveform.shape) < lengths.unsqueeze(
                -1
            )  # (*, L) < (*, 1) = (*, L)
            masked_waveform = waveform * mask
            masked_noise = noise * mask
        else:
            masked_waveform = waveform
            masked_noise = noise

        energy_signal = torch.linalg.vector_norm(masked_waveform, ord=2, dim=-1) ** 2  # (*,)
        energy_noise = torch.linalg.vector_norm(masked_noise, ord=2, dim=-1) ** 2  # (*,)
        original_snr_db = 10 * (torch.log10(energy_signal) - torch.log10(energy_noise))
        scale = 10 ** ((original_snr_db - snr) / 20.0)  # (*,)
        # scale noise
        scaled_noise = scale.unsqueeze(-1) * noise  # (*, 1) * (*, L) = (*, L)
        return waveform + scaled_noise  # (*, L)
    
    
    def compute_per(self, adv_audios: torch.Tensor, ori_audios: torch.Tensor):
        if self.att_norm == 'l2':
            curr_dist = self.mse_loss(
                self.flatten(adv_audios),
                self.flatten(ori_audios)
            )
        elif self.att_norm == 'linf':
            curr_dist = (self.flatten(adv_audios) - self.flatten(ori_audios)).max(1)[0]
        else:
            raise NotImplementedError
        return curr_dist  
    

    def compute_batch_score(self, input_features: torch.Tensor):
        b_size = input_features.shape[0] # batch size
        index_list = [i * self.num_beams for i in range(b_size + 1)]
        pred_len, seqs, out_scores = self.inference(input_features)
        scores = [[] for _ in range(b_size)]
        for out_s in out_scores:
            for i in range(b_size):
                current_index = index_list[i]
                scores[i].append(out_s[current_index: current_index + 1])
        scores = [torch.cat(s) for s in scores]
        scores = [s[:pred_len[i]] for i, s in enumerate(scores)]
        return scores, seqs, pred_len
    
    def compute_score(self, input_features: torch.Tensor, b_size: int = None):
        total_size = input_features.shape[0] # batch size
        if b_size is None:
            b_size = total_size

        if b_size < total_size:
            scores, seqs, pred_len = [], [], []
            for st in range(0, total_size, b_size):
                end = min(st + b_size, total_size)
                score, seq, p_len = self.compute_batch_score(input_features[st: end])
                pred_len.extend(p_len)
                seqs.extend(seq)
                scores.extend(score)
        else:
            scores, seqs, pred_len = self.compute_batch_score(input_features)
        return scores, seqs, pred_len
    
    
    def leave_eos_target_loss(
            self, 
            scores: List[torch.Tensor], 
            seqs: list, 
            pred_lens: list,
        ):
        loss = []
        for i, s in enumerate(scores): # s: T X V
            # print("s {}, seqs[i] {}, pred_lens[i] {}".format(s.shape, seqs[i], pred_lens[i]))
            if pred_lens[i] == 0:
                loss.append(torch.tensor(0.0, requires_grad=True).to(self.device))
            else:
                s[:, self.pad_token_id] = 1e-12
                softmax_v = self.softmax(s)
                eos_p = softmax_v[:pred_lens[i], self.eos_token_id]
                target_p = torch.stack([softmax_v[idx, v] for idx, v in enumerate(seqs[i][1:])])
                target_p = target_p[:pred_lens[i]]
                pred = eos_p + target_p
                pred[-1] = pred[-1] / 2
                loss.append(self.bce_loss(pred, torch.zeros_like(pred)))
        return loss
    

    def data_UE(self, scores: List[torch.Tensor]):
        logits = scores[0].detach().unsqueeze(0) 
        # Mask special tokens probs
        logits[:, :, self.tokenizer.all_special_ids] = - float('inf')
        probs = F.softmax(logits.float(), dim=-1)  # B X T X C
        # ue = data_uncertainty(probs, self.uncertainty_args.data_ue_type)  # B
        entropy = data_uncertainty(probs, 'entropy')  # B
        vanilla = data_uncertainty(probs, 'vanilla')  # B
        return entropy, vanilla
    
    def model_UE(self, input_features: torch.Tensor):
        logger.info("*******Perform stochastic inference*******")
        convert_dropouts(self.model, self.uncertainty_args)
        activate_mc_dropout(self.model, activate=True, random=self.uncertainty_args.inference_prob)

        # self.model.eval()
        # Model uncertainty: MC Dropout stochastic inference
        dropout_eval_results = {}
        dropout_eval_results["sampled_probabilities"] = []
        with torch.no_grad():
            for _ in range(self.uncertainty_args.committee_size):
                scores, seqs, pred_len = self.compute_score(input_features)
                # print("scores {}".format(scores))
                logits = scores[0].unsqueeze(0)
                # Mask special tokens probs
                logits[:, :, self.tokenizer.all_special_ids] = - float('inf')
                probs = F.softmax(logits.float(), dim=-1)  # B X T X C
                dropout_eval_results["sampled_probabilities"].append(probs.tolist())  # K X B X T X C

        # Uncertainty estimation
        prob_array = np.array(dropout_eval_results["sampled_probabilities"])  # K X B X T X C
        if self.model_uncertainty == "bald":
            ue = bald(prob_array)  # B
        elif self.model_uncertainty == "max_prob":
            ue = sampled_max_prob(prob_array)  # B
        elif self.model_uncertainty == "prob_variance":
            ue = probability_variance(prob_array)  # B
        else:
            raise ValueError("Uncertainty type not supported!")

        activate_mc_dropout(self.model, activate=False)
        logger.info("*******Done!!!*******")
        return ue

    
    def compute_loss(self, input_features: torch.Tensor):
        scores, seqs, pred_len = self.compute_score(input_features) # scores: list B of T X V
        loss_list = self.leave_eos_target_loss(scores, seqs, pred_len)
        return loss_list, scores, seqs, pred_len
    
    def run_attack(self, audio: np.ndarray, sample_rate: int = 16000):
        torch.autograd.set_detect_anomaly(True)
        ori_feature = self.process(audio, sample_rate) # (1, L, E)

        with torch.no_grad():
            ori_scores, _, ori_len = self.compute_score(ori_feature)
        
        ori_entropy, ori_vanilla = self.data_UE(ori_scores)
        # ori_mu = self.model_UE(ori_feature)
        ue_dict = [{
            "entropy": ori_entropy.item(), 
            "vanilla": ori_vanilla.item(),
            # "mu": ori_mu.item(), 
            "feature": ori_feature.detach(),
            "pred_len": ori_len,
        }]

        best_adv, best_len = ori_feature.clone(), ori_len
        noise, noise_rate = librosa.load(self.noise, sr=sample_rate)
        noi_feature = self.process(noise, noise_rate) # (1, L, E)
        
        # Handle noise
        if noi_feature.shape[1] < ori_feature.shape[1]:
            K = ceil(ori_feature.shape[1] / noi_feature.shape[1])
            w: torch.Tensor = noi_feature.repeat(1, K, 1)[:, :ori_feature.shape[1], :]
        else:
            w: torch.Tensor = noi_feature[:, :ori_feature.shape[1], :]
        
        w = w.detach()
        # snr_dbs = torch.tensor([20, 10, 3]).to(self.device)
        # w = self.inverse_tanh_space(noi_feature).detach()
        w.requires_grad = True
        optimizer = Adam([w], lr=self.lr)
        pbar = tqdm(range(self.max_iter))
        
        for it in pbar:
            # print("w ({}): {}".format(w.shape, w))
            # adv_audios = self.add_noise(ori_audios, w, snr_dbs)[1:2]
            adv_feature = ori_feature + w
            # print("adv_feature ({}): {}".format(adv_feature.shape, adv_feature))
            # adv_audios = self.tanh_space(w)
            # adv_audios = w
            loss_list, scores, seqs, curr_len = self.compute_loss(adv_feature)
            curr_pred = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in seqs]
            print("curr_pred: {}".format(curr_pred))

            # Calculate uncertainty
            entropy, vanilla = self.data_UE(scores)
            # mu = self.model_UE(adv_feature)
            ue_dict.append({
                "entropy": entropy.item(),
                "vanilla": vanilla.item(), 
                # "mu": mu.item(), 
                "feature": adv_feature.detach(),
                "pred_len": curr_len,
            })
            loss = sum(loss_list)
            # curr_per = self.compute_per(w, ori_audios)
            # print("curr_per", curr_per)
            # per_loss = self.relu(curr_per - self.max_per).sum()
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("w (after): ", w)
            # Update adversarial audios
            # curr_pred, curr_len = self.get_ASR_strings(adv_feature)

            is_best_adv = (curr_len > best_len)
            # mask = torch.tensor((1 - is_best_adv)).to(self.device) * (curr_per < self.max_per)
            # print("mask", mask)
            # mask = mask.view([-1] + [1] * (dim - 1))
            # best_adv = mask * adv_audios.detach() + (1 - mask) * best_adv
            # print("best_adv", best_adv)
            # mask = mask.reshape([-1]).detach().cpu().numpy()
            # best_len = mask * curr_len + (1 - mask) * best_len
            # log_str = "i:%d, curr_len/orig_len:%.2f, per:%.2f, adv_loss:%.2f, per_loss:%.2f" % (
            #     it, float(best_len.sum()) / float(ori_len.sum()), curr_per.mean(), loss, per_loss
            # )
            if is_best_adv:
                best_adv = adv_feature.detach()
                best_len = curr_len
            log_str = "epoch:%d, adv_loss:%.2f, best_len: %.2f, curr_len:%d, entropy:%.4f, vanilla:%.4f" % (
                it, loss, float(sum(best_len))/float(sum(ori_len)), sum(curr_len), entropy[0], vanilla[0],
            )
            pbar.set_description(log_str)
            
        return best_adv, best_len, ue_dict
            
            
        
        
    
    