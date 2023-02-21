from tqdm import tqdm
from math import ceil
import torch
from torch.optim import Adam
from torchaudio.utils import download_asset
from typing import Optional, List
from .ASRbase import BaseAttacker
from utils import SAMPLE_NOISE
from speechbrain.pretrained import EncoderDecoderASR


class ASRSlowAttacker(BaseAttacker):
    
    def __init__(
        self, 
        device: torch.device = None, 
        model: EncoderDecoderASR = None, 
        lr: float = 1e-3,
        max_len: int = 128, 
        max_per: int = None,
        max_iter: int = 5,
        att_norm: str = 'l2',
        noise_file: str = None,
    ):
        super(ASRSlowAttacker, self).__init__(device, model, max_len, lr, max_per, max_iter, att_norm)
        if noise_file is not None:
            noise = download_asset(noise_file)
        else:
            noise = download_asset(SAMPLE_NOISE)
        self.noise = self.model.load_audio(noise).unsqueeze(0).to(self.device)
    
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
    
    def compute_per(self, adv_audios, ori_audios):
        if self.att_norm == 'l2':
            curr_dist = self.mse_Loss(
                self.flatten(adv_audios),
                self.flatten(ori_audios)
            ).sum(dim=1)
        elif self.att_norm == 'linf':
            curr_dist = (self.flatten(adv_audios) - self.flatten(ori_audios)).max(1)[0]
        else:
            raise NotImplementedError
        return curr_dist
    
    def handle_score(self, scores: List[torch.Tensor], pred_lens: List[int], audios: torch.Tensor):
        batch_size = audios.shape[0] # B
        index_list = [i * self.beam_size for i in range(batch_size + 1)]
        out_scores = [[] for _ in range(batch_size)]
        
        for out_s in scores:
            for i in range(batch_size):
                current_index = index_list[i]
                out_scores[i].append(out_s[current_index: current_index + 1])
        out_scores = [torch.cat(s) for s in out_scores]
        out_scores = [s[:pred_lens[i]] for i, s in enumerate(out_scores)]
        return out_scores
    
    
    def leave_eos_target_loss(self, scores: List[torch.Tensor], seqs: list, pred_lens: list):
        loss = []
        for i, s in enumerate(scores): # s: T X V
            # print("s {}, seqs[i] {}, pred_lens[i] {}".format(s.shape, seqs[i], pred_lens[i]))
            if pred_lens[i] == 0:
                loss.append(torch.tensor(0.0, requires_grad=True).to(self.device))
            else:
                s[:, self.pad_token_id] = 1e-12
                softmax_v = self.softmax(s)
                eos_p = softmax_v[:pred_lens[i], self.eos_token_id]
                target_p = torch.stack([softmax_v[idx, v] for idx, v in enumerate(seqs[i])])
                target_p = target_p[:pred_lens[i]]
                pred = eos_p + target_p
                pred[-1] = pred[-1] / 2
                loss.append(self.bce_loss(pred, torch.zeros_like(pred)))
        return loss
    
    
    def compute_loss(self, audios: torch.Tensor, wav_lens: torch.Tensor):
        seqs, _, scores = self.get_predictions(audios, wav_lens)
        pred_lens = [len(seq) for seq in seqs]
        scores = self.handle_score(scores, pred_lens, audios)
        loss_list = self.leave_eos_target_loss(scores, seqs, pred_lens)
        # loss = self.bce_loss(scores, torch.zeros_like(scores))
        return sum(loss_list)
    
    def run_attack(self, audios: torch.Tensor, wav_lens: torch.Tensor):
        dim = len(audios.shape)
        ori_audios = audios.clone()
        ori_len = self.get_ASR_len(audios, wav_lens)
        best_adv = audios.clone()
        best_len = ori_len
        
        # Handle noise
        if self.noise.shape[1] < audios.shape[1]:
            K = ceil(audios.shape[1] / self.noise.shape[1])
            w = self.noise.repeat(1, K)[:, :audios.shape[1]]
        else:
            w = self.noise[:, :audios.shape[1]]
        
        w = w.detach()
        print("w", w)
        snr_dbs = torch.tensor([20, 10, 3]).to(self.device)
        # w = self.inverse_tanh_space(audios).detach()
        w.requires_grad = True
        optimizer = Adam([w], lr=self.lr)
        pbar = tqdm(range(self.max_iter))
        
        for it in pbar:
            print("w: ", w)
            # adv_audios = self.add_noise(ori_audios, w, snr_dbs)[1:2]
            # adv_audios = self.tanh_space(w)
            adv_audios = w + ori_audios 
            loss = self.compute_loss(adv_audios, wav_lens)
            print("loss", loss)
            # curr_per = self.compute_per(adv_audios, ori_audios)
            # print("curr_per", curr_per)
            # per_loss = self.relu(curr_per - self.max_per).sum()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print("w (after)", w)
            
            # Update adversarial audios
            curr_len = self.get_ASR_len(adv_audios, wav_lens)
            is_best_adv = (curr_len > best_len)
            # mask = torch.tensor((1 - is_best_adv)).to(self.device) * (curr_per.detach() < self.max_per)
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
                best_adv = adv_audios.detach()
                best_len = curr_len
                
            log_str = "i:%d, curr_len/orig_len:%.2f, adv_loss:%.2f" % (
                it, float(best_len.sum()) / float(ori_len.sum()), loss,
            )
            
            pbar.set_description(log_str)
        return True, [ori_audios, ori_len], [best_adv, best_len]
            
            
        
        
        
         
    
    