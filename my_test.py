import os
import sys
sys.dont_write_bytecode = True
import argparse
from tqdm import tqdm
import time
import jiwer
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoFeatureExtractor,
    AutoProcessor,
    AutoModelForSpeechSeq2Seq, # s2t, wav2vec2
)
from datasets import load_dataset
from attackers import ASRSlowAttacker


def main(args):
    # Get variables
    lr = args.lr
    max_iter = args.max_iter
    max_len = args.max_len
    att_norm = args.att_norm
    model_n_or_path = args.model
    data_n_or_path = args.dataset
    committee_size = args.committee_size
    data_uncertainty = args.data_uncertainty
    model_uncertainty = args.model_uncertainty
    output_dir = args.output_dir
    device = args.device

    if device == "cuda":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device("cpu")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_n_or_path}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(f"facebook/{model_n_or_path}")
    processor = AutoProcessor.from_pretrained(f"facebook/{model_n_or_path}")
    ds = load_dataset(f"hf-internal-testing/{data_n_or_path}", "clean", split="validation")
    sample_rate = ds[0]["audio"]["sampling_rate"]
    print(model)
    print(ds)

    # Define attacker
    attacker = ASRSlowAttacker(
        device=dev,
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
    
    # Iterate test audios
    pred_file = "{}/{}-{}.pred.txt".format(output_dir, data_n_or_path, model_n_or_path)
    log_file = "{}/{}-{}-{}.log".format(output_dir, data_n_or_path, model_n_or_path, device)
    f = open(pred_file, 'w')
    res = []
    
    time1 = time.time()
    os.system(f"sudo tegrastats --logfile {log_file} &") # start logging energy

    # Inference
    for i, ins in tqdm(enumerate(ds)):
        audio = ins["audio"]['array'] # np.ndarray (seq_len, )
        pred_len, seqs, _ = attacker.get_predictions(audio, sample_rate)
        transcription = processor.batch_decode(seqs, skip_special_tokens=True)[0]
        res.append(transcription.upper())

    os.system("sudo pkill tegrastats") # stop logging energy
    time2 = time.time()
    
    # Write to file
    for output in res:
        f.write("{}\n".format(output))
    
    # Calculate WER
    wer_score = jiwer.wer(ds["text"], res)
    print("Inferencing finished!")
    summary = "Total #seqs: {}, latency: {}s, avg latency: {:.4f}s, WER: {:.4f}".format(
        len(ds), 
        time2 - time1,
        (time2 - time1) / len(ds) if len(ds) > 0 else 0,
        wer_score,
    )
    print(summary)
    
    

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
    parser.add_argument("--output_dir", type=str,
                        default="results",
                        help="Directory to save the results")
    parser.add_argument("--device", "-dev", type=str,
                        default="cuda",
                        choices=[
                            "cpu", 
                            "cuda",
                        ],
                        help="Device to run the model on.")
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