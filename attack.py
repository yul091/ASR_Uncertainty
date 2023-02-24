import sys
sys.dont_write_bytecode = True
import torch
import argparse
# from models.speechbrain_pretrained import MyEncoderDecoderASR
# from speechbrain.pretrained import EncoderASR
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoProcessor,
    Speech2TextProcessor, 
    AutoModelForCTC,
    AutoModelForSpeechSeq2Seq,
    SpeechEncoderDecoderModel, # wav2vec2
    Speech2TextForConditionalGeneration,
    AutoModelForCTC,
    HubertForCTC,
    SEWForCTC,
    SEWDForCTC,
)
from datasets import load_dataset
from attackers.ASRSlow import ASRSlowAttacker


def main(args: argparse.Namespace):
    max_iter = args.max_iter
    lr = args.lr
    att_norm = args.att_norm
    model_n_or_path = args.model
    dataset_n_or_path = args.dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForSpeechSeq2Seq.from_pretrained(f"facebook/{model_n_or_path}")
    processor = AutoProcessor.from_pretrained(f"facebook/{model_n_or_path}")
    ds = load_dataset(f"hf-internal-testing/{dataset_n_or_path}", "clean", split="validation")
    print(model)
    print(ds)

    print("input: {}".format(ds[0]["audio"]["array"].shape))
    inputs = processor(
        ds[0]["audio"]["array"], 
        sampling_rate=ds[0]["audio"]["sampling_rate"], 
        return_tensors="pt",
    ) # input_features, attention_mask
    input_features = inputs.input_features
    print("input feature: {}".format(input_features.shape))
    
    model = model.to(device)
    input_features = input_features.to(device)
    
    generated_ids = model.generate(
        inputs=input_features,
        max_length=128,
    )
    print("generated_ids ({}): {}".format(generated_ids.shape, generated_ids))
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("output: {}".format(transcription))
    
        
    # # Define attacker
    # attacker = ASRSlowAttacker(
    #     device=device,
    #     model=model, 
    #     lr=lr,
    #     att_norm=att_norm,
    #     max_iter=max_iter,
    # )
    
    # # asr_model.transcribe_file(audio_1)
    # waveform = asr_model.load_audio(audio).unsqueeze(0).to(device)
    # wav_lens = torch.tensor([1.0]).to(device)
    # pred_words, pred_tokens, scores = asr_model.transcribe_batch(waveform, wav_lens)
    # encoder_out = asr_model.encode_batch(waveform, wav_lens) # B X T X D
    # print("encoder outputs: ", encoder_out.shape)
    # predicted_tokens, topk_scores, scores = asr_model.mods.decoder(encoder_out, wav_lens)
    # print(predicted_tokens)
    # print("topk_scores ({}): {}".format(topk_scores.shape, topk_scores))
    # print("scores ({})".format([score.shape for score in scores]))
    # predicted_words = [
    #     asr_model.tokenizer.decode_ids(token_seq)
    #     for token_seq in predicted_tokens
    # ]
    # print("orig outputs: ", pred_words)
    
    # # Attack
    # Success, [ori_audios, ori_len], [best_adv, best_len] = attacker.run_attack(
    #     audios=waveform,
    #     wav_lens=wav_lens,
    # )
    # print("best_len: {}, best_adv: {}".format(best_len, best_adv))
    # pred_words, pred_tokens, scores = model.transcribe_batch(best_adv, wav_lens)
    # print(pred_words)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="s2t-small-librispeech-asr",
                        choices=[
                            "s2t-small-librispeech-asr",
                        ],
                        help="Path to the victim model")
    parser.add_argument("--dataset", type=str,
                        default="librispeech_asr_dummy",
                        help="Dataset to use for testing")
    parser.add_argument("--max_iter", type=int,
                        default=5,
                        help="Maximum number of iterations")
    parser.add_argument("--lr", type=float,
                        default=0.01,
                        help="Learning rate")
    parser.add_argument("--att_norm", type=str,
                        default='l2',
                        choices=['l2', 'linf'],
                        help="Norm to use for the attack")
    
    args = parser.parse_args()
    main(args)

    