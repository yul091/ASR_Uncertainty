import sys
sys.dont_write_bytecode = True
import torch
import argparse
# from models.speechbrain_pretrained import MyEncoderDecoderASR
# from speechbrain.pretrained import EncoderASR
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoFeatureExtractor,
    AutoProcessor,
    AutoModelForSpeechSeq2Seq, # s2t, wav2vec2
)
from datasets import load_dataset
from attackers.ASRSlow import ASRSlowAttacker


def main(args: argparse.Namespace):
    lr = args.lr
    max_iter = args.max_iter
    max_len = args.max_len
    att_norm = args.att_norm
    model_n_or_path = args.model
    data_n_or_path = args.dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_n_or_path}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(f"facebook/{model_n_or_path}")
    processor = AutoProcessor.from_pretrained(f"facebook/{model_n_or_path}")
    ds = load_dataset(f"hf-internal-testing/{data_n_or_path}", "clean", split="validation")
    print(model)
    print(ds)

    audio = ds[0]["audio"]["array"]
    sample_rate = ds[0]["audio"]["sampling_rate"]
    print("input ({}): {}, sample rate: {}".format(type(audio), audio.shape, sample_rate))
        
    # Define attacker
    attacker = ASRSlowAttacker(
        device=device,
        tokenizer=tokenizer,
        processor=processor,
        model=model, 
        lr=lr,
        att_norm=att_norm,
        max_iter=max_iter,
        max_len=max_len,
    )

    # Inference
    pred_len, seqs, _ = attacker.get_predictions(audio, sample_rate)
    transcription = processor.batch_decode(seqs, skip_special_tokens=True)[0]
    print("output ({}): {}".format(pred_len, transcription))
    
    # Attack
    best_adv, best_len = attacker.run_attack(
        audio=audio,
        sample_rate=sample_rate,
    )
    pred_len, best_seqs, _ = attacker.inference(best_adv)
    assert pred_len == best_len
    transcription = processor.batch_decode(best_seqs, skip_special_tokens=True)[0]
    print("best_len: {}, output: {}".format(best_len, transcription))


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
                        default=0.001,
                        help="Learning rate")
    parser.add_argument("--att_norm", type=str,
                        default='l2',
                        choices=['l2', 'linf'],
                        help="Norm to use for the attack")
    parser.add_argument("--max_len", type=int,
                        default=128,
                        help="Maximum length of sequence to generate")
    
    
    args = parser.parse_args()
    main(args)

    