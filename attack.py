import sys
sys.dont_write_bytecode = True
import torch
from speechbrain.pretrained import EncoderDecoderASR
from attackers.ASRSlow import ASRSlowAttacker




if __name__ == "__main__":

    audio = "LibriSpeech/test-clean/1089/134686/1089-134686-0030.flac"
    model = "asr-crdnn-rnnlm-librispeech" # asr-crdnn-transformerlm-librispeech, asr-transformer-transformerlm-librispeech
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_iter = 5
    lr = 0.001
    att_norm = 'l2'
    
    # Uncomment for using another pre-trained model
    asr_model = EncoderDecoderASR.from_hparams(
        source=f"speechbrain/{model}", 
        savedir=f"pretrained_models/{model}", 
        run_opts={"device":"cuda"}, # inference on GPU
        hparams_file="hyperparams.yaml",
    )
    # Define attacker
    attacker = ASRSlowAttacker(
        device=device,
        model=asr_model, 
        lr=lr,
        att_norm=att_norm,
        max_iter=max_iter,
    )
    
    # asr_model.transcribe_file(audio_1)
    waveform = asr_model.load_audio(audio).unsqueeze(0).to(device)
    wav_lens = torch.tensor([1.0]).to(device)
    # predicted_words, predicted_tokens = asr_model.transcribe_batch(batch, rel_length)
    with torch.no_grad():
        encoder_out = asr_model.encode_batch(waveform, wav_lens) # B X T X D
        print("encoder outputs: ", encoder_out.shape)
        predicted_tokens, topk_scores, scores = asr_model.mods.decoder(encoder_out, wav_lens)
        print(predicted_tokens)
        print("topk_scores ({}): {}".format(topk_scores.shape, topk_scores))
        print("scores ({})".format([score.shape for score in scores]))
        predicted_words = [
            asr_model.tokenizer.decode_ids(token_seq)
            for token_seq in predicted_tokens
        ]
        print("orig outputs: ", predicted_words)
        
        # Attack
        Success, [ori_audios, ori_len], [best_adv, best_len] = attacker.run_attack(
            audios=waveform,
            wav_lens=wav_lens,
        )
        print("best_len: {}, best_adv: {}".format(best_len, best_adv))
        predicted_words, predicted_tokens = asr_model.transcribe_batch(best_adv, wav_lens)
        print(predicted_words)