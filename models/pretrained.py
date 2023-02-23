import torch
from speechbrain.pretrained import EncoderDecoderASR


class MyEncoderDecoderASR(EncoderDecoderASR):
    """A ready-to-use Encoder-Decoder ASR model

    The class can be used either to run only the encoder (encode()) to extract
    features or to run the entire encoder-decoder model
    (transcribe()) to transcribe speech. The given YAML must contains the fields
    specified in the *_NEEDED[] lists.

    Example
    -------
    >>> from speechbrain.pretrained import EncoderDecoderASR
    >>> tmpdir = getfixture("tmpdir")
    >>> asr_model = EncoderDecoderASR.from_hparams(
    ...     source="speechbrain/asr-crdnn-rnnlm-librispeech",
    ...     savedir=tmpdir,
    ... )
    >>> asr_model.transcribe_file("tests/samples/single-mic/example2.flac")
    "MY FATHER HAS REVEALED THE CULPRIT'S NAME"
    """

    HPARAMS_NEEDED = ["tokenizer"]
    MODULES_NEEDED = ["encoder", "decoder"]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = self.hparams.tokenizer
        
    def transcribe_file(self, path):
        """Transcribes the given audiofile into a sequence of words.

        Arguments
        ---------
        path : str
            Path to audio file which to transcribe.

        Returns
        -------
        str
            The audiofile transcription produced by this ASR system.
        """
        waveform = self.load_audio(path)
        # Fake a batch:
        batch = waveform.unsqueeze(0)
        rel_length = torch.tensor([1.0])
        predicted_words, predicted_tokens, scores = self.transcribe_batch(
            batch, rel_length
        )
        return predicted_words[0]
    
    def transcribe_batch(self, wavs, wav_lens):
        """Transcribes the input audio into a sequence of words

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = EncoderDecoderASR.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model.
        wav_lens : torch.tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        list
            Each waveform in the batch transcribed.
        tensor
            Each predicted token id.
        """
        with torch.no_grad():
            wav_lens = wav_lens.to(self.device)
            encoder_out = self.encode_batch(wavs, wav_lens)
            predicted_tokens, topk_scores, scores = self.mods.decoder(encoder_out, wav_lens)
            predicted_words = [
                self.tokenizer.decode_ids(token_seq)
                for token_seq in predicted_tokens
            ]
        return predicted_words, predicted_tokens, scores
    
    def encode_batch(self, wavs, wav_lens):
        """Encodes the input audio into a sequence of hidden states

        The waveforms should already be in the model's desired format.
        You can call:
        ``normalized = EncoderDecoderASR.normalizer(signal, sample_rate)``
        to get a correctly converted signal in most cases.

        Arguments
        ---------
        wavs : torch.tensor
            Batch of waveforms [batch, time, channels] or [batch, time]
            depending on the model.
        wav_lens : torch.tensor
            Lengths of the waveforms relative to the longest one in the
            batch, tensor of shape [batch]. The longest one should have
            relative length 1.0 and others len(waveform) / max_length.
            Used for ignoring padding.

        Returns
        -------
        torch.tensor
            The encoded batch
        """
        print("Calling self-defined encode_batch() !")
        wavs = wavs.float()
        encoder_out = self.mods.encoder(wavs, wav_lens)
        return encoder_out

    
    
if __name__ == "__main__":
    import torch

    audio = "LibriSpeech/test-clean/1089/134686/1089-134686-0030.flac"
    model = "asr-crdnn-rnnlm-librispeech" # asr-crdnn-transformerlm-librispeech, asr-transformer-transformerlm-librispeech

    # Uncomment for using another pre-trained model
    asr_model = MyEncoderDecoderASR.from_hparams(
        source=f"speechbrain/{model}", 
        savedir=f"pretrained_models/{model}", 
        # run_opts={"device":"cuda"}, # inference on GPU
        hparams_file="hyperparams.yaml",
    )
    # asr_model.transcribe_file(audio_1)
    waveform = asr_model.load_audio(audio)
    # Fake a batch
    batch = waveform.unsqueeze(0)
    print('wave tensor: ', batch.shape)
    rel_length = torch.tensor([1.0])
    # predicted_words, predicted_tokens = asr_model.transcribe_batch(batch, rel_length)
    with torch.no_grad():
        wav_lens = rel_length
        encoder_out = asr_model.encode_batch(batch, wav_lens) # B X T X D
        print("encoder output: ", encoder_out.shape)
        predicted_tokens, topk_scores, scores = asr_model.mods.decoder(encoder_out, wav_lens)
        print(predicted_tokens)
        print("topk_scores ({}): {}".format(topk_scores.shape, topk_scores))
        print("beam scores ({})".format([s.shape for s in scores]))
        predicted_words = [
            asr_model.tokenizer.decode_ids(token_seq)
            for token_seq in predicted_tokens
        ]
        print("predictions: ", predicted_words)