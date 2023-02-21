# import os
# from pathlib import Path
# SAMPLE_RATE = 44100
# BASE_DIR = Path(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
# SCRIPTS_DIR = BASE_DIR / "scripts"
# TEST_FIXTURES_DIR = BASE_DIR / "test_fixtures"
import sys
sys.dont_write_bytecode = True
import torch
from uncertainty import data_uncertainty
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.decoders import S2SRNNBeamSearchLM
from torch_audiomentations import (
    Compose, 
    Gain, 
    PolarityInversion, 
    AddBackgroundNoise,
)


model = "asr-crdnn-rnnlm-librispeech" # asr-crdnn-transformerlm-librispeech, asr-transformer-transformerlm-librispeech
# Uncomment for using another pre-trained model
asr_model = EncoderDecoderASR.from_hparams(
    source=f"speechbrain/{model}", 
    savedir=f"pretrained_models/{model}", 
    # run_opts={"device":"cuda"}, # inference on GPU
    # return_log_probs=True,
)
print(asr_model)
# asr_model.transcribe_file(audio_1)

SAMPLE_WAV = "LibriSpeech/test-clean/1089/134686/1089-134686-0030.flac"
waveform = asr_model.load_audio(SAMPLE_WAV)
batch = waveform.unsqueeze(0)

# Initialize augmentation callable
apply_augmentation = Compose(
    transforms=[
        Gain(
            min_gain_in_db=-15.0,
            max_gain_in_db=5.0,
            p=0.5,
        ),
        PolarityInversion(p=0.5),
    ]
)

perturbed_audio_samples = apply_augmentation(batch.unsqueeze(0), sample_rate=16000)

# from IPython.display import Audio
# Audio(perturbed_audio_samples.squeeze(0), rate=16000)


rel_length = torch.tensor([1.0])
# predicted_words, predicted_tokens = asr_model.transcribe_batch(batch, rel_length)
with torch.no_grad():
    wav_lens = rel_length
    encoder_out = asr_model.encode_batch(batch, wav_lens) # B X T X D
    predicted_tokens, scores = asr_model.mods.decoder(encoder_out, wav_lens)
    print(predicted_tokens) # list of B lists of T tokens
    print(scores) # B X T X V
    predicted_words = [
        asr_model.tokenizer.decode_ids(token_seq)
        for token_seq in predicted_tokens
    ]
    print(predicted_words) # list of B strings