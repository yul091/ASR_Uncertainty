# pip install speechbrain
# pip install evaluate
# pip install jiwer 
import os
import sys
sys.dont_write_bytecode = True
import shutil
import argparse
from evaluate import load
from tqdm import tqdm
from speechbrain.utils.data_utils import download_file
from speechbrain.pretrained import EncoderDecoderASR
# from speechbrain.utils.metric_stats import ErrorRateStats
import time
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence


def main(args):
    # Get variables
    model_name = args.model
    dataset = args.dataset
    split = "test-clean"

    # Load from pre-trained model
    asr_model = EncoderDecoderASR.from_hparams(
        source=f"speechbrain/{model_name}", 
        savedir=f"pretrained_models/{model_name}", 
        run_opts={"device":"cuda"}, # inference on GPU
    )
    print(asr_model)
    
    ###################################### Download #######################################
    # Download + Unpacking test-clean of librispeech
    if not os.path.exists(dataset):
        os.makedirs(dataset)
        MINILIBRI_TEST_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
        download_file(MINILIBRI_TEST_URL, 'test-clean.tar.gz')
        shutil.unpack_archive( 'test-clean.tar.gz', '.')
    #######################################################################################
    
    # Load metrics
    wer = load('wer')
    
    # Iterate test audios
    user = 61
    para = 70968
    extension = ".flac" # .wav
    audio_folder = "{}/{}/{}/{}".format(dataset, split, user, para)
    dialog_list = sorted(os.listdir(audio_folder))
    pred_file = "{}-{}-{}-{}.preds.txt".format(dataset, user, para, model_name)
    label_file = "{}/{}/{}/{}/{}-{}.trans.txt".format(dataset, split, user, para, user, para)
    f = open(pred_file, 'w')
    count = 0
    res = []
    
    time1 = time.time()
    os.system(f"sudo tegrastats --logfile {dataset}-{model_name}-energy.log &") # start logging energy
    for audio in tqdm(dialog_list):
        if not audio.endswith(extension):
            continue
        count += 1
        audio_path = os.path.join(audio_folder, audio)
        output: str = asr_model.transcribe_file(audio_path)
        res.append((audio, output.upper()))
    os.system("sudo pkill tegrastats") # stop logging energy
    time2 = time.time()
    
    # Write to file
    for (audio, output) in res:
        f.write("{} {}\n".format(audio.rstrip(extension), output))
    
    # Calculate WER
    # f1 = open(pred_file, "r")
    f2 = open(label_file, "r")
    pred_res = [output for (audio, output) in res]
    label_res = [' '.join(line.split()[1:]) for line in f2.read().splitlines()]
    wer_score = wer.compute(predictions=pred_res, references=label_res)
    # print(wer_score)
    summary = "Inference finished! Total #seqs: {}, latency: {}s, avg latency: {:.4f}s, WER: {:.4f}".format(
        count, 
        time2 - time1,
        (time2 - time1) / count if count > 0 else 0,
        wer_score,
    )
    print(summary)
    
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, 
                        default="asr-crdnn-rnnlm-librispeech", 
                        choices=[
                            "asr-crdnn-rnnlm-librispeech",
                            "asr-crdnn-transformerlm-librispeech",
                            "asr-transformer-transformerlm-librispeech",
                        ],
                        help="Path to the victim model")
    parser.add_argument("--dataset", "-d", type=str,
                        default="LibriSpeech",
                        choices=[
                            "LibriSpeech",
                            "TIMIT",
                        ],
                        help="Dataset to test for speech-to-text WER.")
    args = parser.parse_args()
    main(args)

    # ################################# Decode in the batch #################################
    # # Decode the first sentence in the batch
    # snt_1, fs = torchaudio.load(audio_1)
    # wav_lens=torch.tensor([1.0])
    # asr_model.transcribe_batch(snt_1, wav_lens)

    # # Decode another sentence in the batch
    # audio_2 = "/content/LibriSpeech/test-clean/1089/134686/1089-134686-0007.flac"
    # snt_2, fs = torchaudio.load(audio_2)
    # wav_lens=torch.tensor([1.0])
    # asr_model.transcribe_batch(snt_2, wav_lens)

    # # Decode both sentences within the same batch
    # # Padding
    # batch = pad_sequence(
    #     [snt_1.squeeze(), snt_2.squeeze()], 
    #     batch_first=True, 
    #     padding_value=0.0,
    # )
    # wav_lens=torch.tensor([snt_1.shape[1]/batch.shape[1], snt_2.shape[1]/batch.shape[1]])
    # asr_model.transcribe_batch(batch, wav_lens)

    ############################ Set up a batch of 8 sentences ############################
    # audio_files=[]
    # audio_files.append('/content/LibriSpeech/test-clean/1089/134686/1089-134686-0030.flac')
    # audio_files.append('/content/LibriSpeech/test-clean/1089/134686/1089-134686-0014.flac')
    # audio_files.append('/content/LibriSpeech/test-clean/1089/134686/1089-134686-0007.flac')
    # audio_files.append('/content/LibriSpeech/test-clean/1089/134691/1089-134691-0000.flac')
    # audio_files.append('/content/LibriSpeech/test-clean/1089/134691/1089-134691-0003.flac')
    # audio_files.append('/content/LibriSpeech/test-clean/1188/133604/1188-133604-0030.flac')
    # audio_files.append('/content/LibriSpeech/test-clean/1089/134691/1089-134691-0019.flac')
    # audio_files.append('/content/LibriSpeech/test-clean/1188/133604/1188-133604-0006.flac')

    # sigs=[]
    # lens=[]
    # for audio_file in audio_files:
    #     snt, fs = torchaudio.load(audio_file)
    #     sigs.append(snt.squeeze())
    #     lens.append(snt.shape[1])

    # batch = pad_sequence(sigs, batch_first=True, padding_value=0.0)
    # lens = torch.Tensor(lens) / batch.shape[1]
    # asr_model.transcribe_batch(batch, lens)