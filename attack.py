import sys
sys.dont_write_bytecode = True
import os
import time
import jiwer
import torch
import argparse
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoFeatureExtractor,
    AutoProcessor,
    AutoModelForSpeechSeq2Seq, # s2t, wav2vec2
)
from datasets import load_dataset
from attackers import ASRSlowAttacker


def main(args: argparse.Namespace):
    lr = args.lr
    max_iter = args.max_iter
    max_len = args.max_len
    att_norm = args.att_norm
    model_n_or_path = args.model
    data_n_or_path = args.dataset
    output_dir = args.output_dir
    committee_size = args.committee_size
    data_uncertainty = args.data_uncertainty
    model_uncertainty = args.model_uncertainty
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(model_n_or_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_n_or_path)
    processor = AutoProcessor.from_pretrained(model_n_or_path)
    ds = load_dataset(f"hf-internal-testing/{data_n_or_path}", "clean", split="validation")
    sample_rate = ds[0]["audio"]["sampling_rate"]
    print(model)
    print(ds)
        
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
        committee_size=committee_size,
        data_uncertainty=data_uncertainty,
        model_uncertainty=model_uncertainty,
    )

    model_n = model_n_or_path.split("/")[-1]
    pred_file = "{}/{}-{}.adv.pred.txt".format(output_dir, data_n_or_path, model_n)
    log_file = "{}/{}-{}-{}.adv.log".format(output_dir, data_n_or_path, model_n, device)

    # Inference & Attack
    if not os.path.exists(f"{output_dir}/ue_dicts.pt"):    
        all_ue_dicts = []
        for i, ins in tqdm(enumerate(ds)):
            audio = ins["audio"]['array'] # np.ndarray (seq_len, )
            pred_len, seqs, _ = attacker.get_predictions(audio, sample_rate)
            transcription = processor.batch_decode(seqs, skip_special_tokens=True)[0]
            print("output ({}): {}".format(pred_len, transcription))
        
            # Attack
            best_adv, best_len, ue_dicts = attacker.run_attack(
                audio=audio,
                sample_rate=sample_rate,
            )
            all_ue_dicts.extend(ue_dicts)
            pred_len, best_seqs, _ = attacker.inference(best_adv)
            assert pred_len == best_len
            transcription = processor.batch_decode(best_seqs, skip_special_tokens=True)[0]
            print("best_len: {}, output: {}".format(best_len, transcription))
            
        torch.save(all_ue_dicts, f"{output_dir}/ue_dicts.pt")
    else:
        all_ue_dicts = torch.load(f"{output_dir}/ue_dicts.pt")

    # Get statistics by iterating adversarial audios
    if not os.path.exists(f"{output_dir}/latency.pt"):
        res, latency = [], []
        os.system(f"sudo tegrastats --logfile {log_file} &") # start logging energy

        # Inference
        for i, ins in tqdm(enumerate(all_ue_dicts)):
            time1 = time.time()
            feature = ins["feature"] # torch.tensor (seq_len, )
            pred_len, seqs, _ = attacker.inference(feature)
            transcription = processor.batch_decode(seqs, skip_special_tokens=True)[0]
            time2 = time.time()
            res.append(transcription.upper())
            latency.append(time2 - time1)

        os.system("sudo pkill tegrastats") # stop logging energy
        torch.save(latency, f"{output_dir}/latency.pt")

        # Write to file
        f = open(pred_file, 'w')
        for output in res:
            f.write("{}\n".format(output))
    else:
        latency = torch.load(f"{output_dir}/latency.pt")
        res = [line.strip() for line in open(pred_file, 'r')]

    
    # Calculate WER
    if not os.path.exists(f"{output_dir}/wer_scores.pt"):
        wer_score = jiwer.wer(np.repeat(ds["text"], 1+max_iter).tolist(), res)
        wer_scores = [jiwer.wer(ds["text"][i//(1+max_iter)], res[i]) for i in range(len(all_ue_dicts))]
        print("Inferencing finished!")
        summary = "Total #seqs: {}, latency: {}s, avg latency: {:.4f}s, avg WER: {:.4f}".format(
            len(ds), 
            sum(latency),
            sum(latency) / len(latency) if len(latency) > 0 else 0,
            wer_score,
        )
        print(summary)
        torch.save(wer_scores, f"{output_dir}/wer_scores.pt")
    else:
        wer_scores = torch.load(f"{output_dir}/wer_scores.pt")
        wer_score = jiwer.wer(np.repeat(ds["text"], 1+max_iter).tolist(), res)
        summary = "Total #seqs: {}, latency: {}s, avg latency: {:.4f}s, avg WER: {:.4f}".format(
            len(ds), 
            sum(latency),
            sum(latency) / len(latency) if len(latency) > 0 else 0,
            wer_score,
        )
        print(summary)
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="facebook/s2t-small-librispeech-asr",
                        choices=[
                            "facebook/s2t-small-librispeech-asr",
                            "openai/whisper-tiny.en",
                        ],
                        help="Path to the victim model")
    parser.add_argument("--dataset", type=str,
                        default="librispeech_asr_dummy",
                        help="Dataset to use for testing")
    parser.add_argument("--output_dir", type=str,
                        default="results",
                        help="Directory to save the results")
    parser.add_argument("--max_iter", type=int,
                        default=10,
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
    parser.add_argument("--committee_size", type=int,
                        default=10,
                        help="Number of stochastic inferences")
    parser.add_argument("--data_uncertainty", '-du', type=str,
                        default='vanilla',
                        choices=['vanilla', 'entropy'],
                        help="Data uncertainty estimation method")
    parser.add_argument("--model_uncertainty", '-mu', type=str,
                        default='prob_variance',
                        choices=['prob_variance', 'bald', 'sampled_max_prob'],
                        help="Model uncertainty estimation method")
    
    args = parser.parse_args()
    main(args)

    