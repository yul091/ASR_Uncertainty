import numpy as np
from tqdm import tqdm
from math import ceil
import torch
import librosa
from torch.optim import Adam
import torchaudio
from torchaudio.utils import download_asset
from typing import Optional, List
from .ASRbase import BaseAttacker
from utils import SAMPLE_NOISE
from transformers import (
    Speech2Text2Tokenizer,
    Speech2Text2Processor,
    Speech2TextForConditionalGeneration,
)


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
    
    def compute_per(
            self, 
            adv_audios: torch.Tensor, 
            ori_audios: torch.Tensor,
        ):
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
    

    def compute_batch_score(
            self, 
            audios: List[np.ndarray], 
            sample_rate: int,
        ):
        batch_size = len(audios) # batch size
        index_list = [i * self.num_beams for i in range(batch_size + 1)]
        pred_len, seqs, out_scores = self.get_predictions(audios, sample_rate)
        print("pred_len {}, seqs {}, out_scores {}".format(pred_len, seqs, [s.shape for s in out_scores]))
        scores = [[] for _ in range(batch_size)]
        for out_s in out_scores:
            for i in range(batch_size):
                current_index = index_list[i]
                scores[i].append(out_s[current_index: current_index + 1])
        scores = [torch.cat(s) for s in scores]
        scores = [s[:pred_len[i]] for i, s in enumerate(scores)]
        return scores, seqs, pred_len
    
    def compute_score(
            self, 
            audios: List[np.ndarray], 
            sample_rate: int, 
            b_size: int = None
        ):
        total_size = len(audios) # batch size
        if b_size is None:
            b_size = total_size

        if b_size < total_size:
            scores, seqs, pred_len = [], [], []
            for st in range(0, total_size, b_size):
                end = min(st + b_size, total_size)
                score, seq, p_len = self.compute_batch_score(audios[st: end, :], sample_rate)
                pred_len.extend(p_len)
                seqs.extend(seq)
                scores.extend(score)
        else:
            scores, seqs, pred_len = self.compute_batch_score(audios, sample_rate)
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
                print("s.shape: {}, self.pad_token_id: {}, self.eos_token_id: {}".format(s.shape, self.pad_token_id, self.eos_token_id))
                s[:, self.pad_token_id] = 1e-12
                softmax_v = self.softmax(s)
                eos_p = softmax_v[:pred_lens[i], self.eos_token_id]
                target_p = torch.stack([softmax_v[idx, v] for idx, v in enumerate(seqs[i][1:])])
                target_p = target_p[:pred_lens[i]]
                pred = eos_p + target_p
                pred[-1] = pred[-1] / 2
                loss.append(self.bce_loss(pred, torch.zeros_like(pred)))
        return loss
    
    def compute_loss(
            self, 
            audios: List[np.ndarray], 
            sample_rate: int,
        ):
        scores, seqs, pred_len = self.compute_score(audios, sample_rate)
        loss_list = self.leave_eos_target_loss(scores, seqs, pred_len)
        # loss = self.bce_loss(scores, torch.zeros_like(scores))
        return loss_list
    
    def run_attack(
            self, 
            audios: List[np.ndarray], 
            sample_rate: int,
        ):
        ori_len = self.get_ASR_len(audios, sample_rate)

        torch.autograd.set_detect_anomaly(True)
        if not isinstance(audios, torch.Tensor):
            audios = torch.tensor(audios).to(self.device)
        if len(audios.shape) == 1:
            audios = audios.unsqueeze(0)
        dim = len(audios.shape)
        
        ori_audios, best_adv = audios.clone(), audios.clone()
        best_len = ori_len

        noise, noise_rate = librosa.load(self.noise, sr=sample_rate)
        noise = torch.tensor(noise).to(self.device)
        if len(noise.shape) == 1:
            noise = noise.unsqueeze(0)

        print("audios ({}): {}".format(audios.shape, audios))
        print("noise ({}): {}".format(noise.shape, noise))
        
        # Handle noise
        if noise.shape[1] < audios.shape[1]:
            K = ceil(audios.shape[1] / noise.shape[1])
            w: torch.Tensor = noise.repeat(1, K)[:, :audios.shape[1]]
        else:
            w: torch.Tensor = noise[:, :audios.shape[1]]
        
        w = w.detach()
        snr_dbs = torch.tensor([20, 10, 3]).to(self.device)
        # w = self.inverse_tanh_space(audios).detach()
        w.requires_grad = True
        optimizer = Adam([w], lr=self.lr)
        pbar = tqdm(range(self.max_iter))
        
        for it in pbar:
            print("w ({}): {}".format(w.shape, w))
            # print("w.grad: {}".format(w.grad))
            # adv_audios = self.add_noise(ori_audios, w, snr_dbs)[1:2]
            adv_audios = ori_audios + w
            print("adv_audios ({}): {}".format(adv_audios.shape, adv_audios))
            # adv_audios = self.tanh_space(w)
            # adv_audios = w
            loss_list = self.compute_loss([adv_audios.squeeze(0).detach().cpu().numpy()], sample_rate)
            loss = sum(loss_list)
            # loss = self.mse_loss(w, torch.randn_like(w)).sum()
            print("loss: {}".format(loss))
            # curr_per = self.compute_per(w, ori_audios)
            # print("curr_per", curr_per)
            # per_loss = self.relu(curr_per - self.max_per).sum()
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print("w (after): ", w)
            # print("w.grad (after): ", w.grad)
            # Update adversarial audios
            # curr_len = self.get_ASR_len(adv_audios, wav_lens)
            # is_best_adv = (curr_len > best_len)
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
            # if is_best_adv:
            #     best_adv = adv_audios.detach()
            #     best_len = curr_len
            # log_str = "i:%d, curr_len/orig_len:%.2f, adv_loss:%.2f" % (
            #     it, float(best_len.sum()) / float(ori_len.sum()), loss,
            # )
            # pbar.set_description(log_str)
            
        return [ori_audios, ori_len], [best_adv.squeeze(0).detach().cpu().numpy(), best_len]
            
            
        
        
        
         
    
    