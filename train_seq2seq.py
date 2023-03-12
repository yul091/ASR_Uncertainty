import os
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
import sys
sys.dont_write_bytecode = True
from transformers import AutoTokenizer

from typing import Union, List, Dict, Tuple, Optional, Any, Sequence
import math
import glob
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import argparse
from argparse import Namespace

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from utils import set_seed
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.dataset import Seq2SeqDataset, Seq2SeqCollator
# from seq2seq.optim import Optimizer
# from seq2seq.evaluator import Predictor
# from seq2seq.util.checkpoint import Checkpoint
from nltk.translate.bleu_score import sentence_bleu


DATA_DIR = 'datasets'
DEFAULT_DATA = {
    'train': os.path.join(DATA_DIR, 'train.tsv'),
    'validation': os.path.join(DATA_DIR, 'val.tsv'),
    'test': os.path.join(DATA_DIR, 'dev.tsv'),
}


class Seq2SeqSystem(pl.LightningModule):
    def __init__(self, hparams: Namespace):
        super(Seq2SeqSystem, self).__init__()
        
        # Define hyperparameters
        self.src_field_name = getattr(hparams, 'src_field_name', 'src')
        self.tgt_field_name = getattr(hparams, 'tgt_field_name', 'tgt')
        self.teacher_forcing_ratio = getattr(hparams, 'teacher_forcing_ratio', 0.5)
        self.decode_function = getattr(hparams, 'decode_function', F.log_softmax)
        self.tok_path = getattr(hparams, 'tok_path', 'bert-base-uncased')
        self.data_files = getattr(hparams, 'data_files', DEFAULT_DATA)
        self.max_len = getattr(hparams, 'max_len', 128)
        self.hidden_size = getattr(hparams, 'hidden_size', 256)
        self.bidirectional = getattr(hparams, 'bidirectional', True)
        self.input_dropout = getattr(hparams, 'input_dropout', 0)
        self.dropout = getattr(hparams, 'dropout', 0.2)
        self.n_layers = getattr(hparams, 'n_layers', 1)
        self.rnn_cell = getattr(hparams, 'rnn_cell', 'gru')
        self.attention = getattr(hparams, 'attention', False)
        
        # Define tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.tok_path)
        encoder = EncoderRNN(
            vocab_size=self.tokenizer.__len__(), 
            max_len=self.max_len, 
            hidden_size=self.hidden_size,
            input_dropout_p=self.input_dropout,
            dropout_p=self.dropout,
            n_layers=self.n_layers,
            bidirectional=self.bidirectional, 
            rnn_cell=self.rnn_cell,
        )
        decoder = DecoderRNN(
            vocab_size=self.tokenizer.__len__(), 
            max_len=self.max_len, 
            hidden_size=self.hidden_size * 2 if self.bidirectional else self.hidden_size,
            sos_id=self.tokenizer.bos_token_id,
            eos_id=self.tokenizer.eos_token_id,
            input_dropout_p=self.input_dropout,
            dropout_p=self.dropout,
            n_layers=self.n_layers,
            bidirectional=self.bidirectional,
            rnn_cell=self.rnn_cell,
            use_attention=self.attention,
        )
        self.model = Seq2seq(encoder, decoder, decode_function=self.decode_function)
        for param in self.model.parameters():
            param.data.uniform_(-0.08, 0.08)
            
        # Define training arguments
        self.base_lr = getattr(hparams, 'base_lr', 1e-4)
        self.n_gpus = getattr(hparams, 'n_gpus', 1)
        self.train_batch_size = getattr(hparams, 'train_batch_size', 32)
        self.eval_batch_size = getattr(hparams, 'eval_batch_size', 32)
        self.fp16 = getattr(hparams, 'fp16', False)
        self.num_workers = getattr(hparams, 'num_workers', 4)
        
        # Data collator
        self.data_collator = Seq2SeqCollator(
            tokenizer=self.tokenizer,
            padding=True,
            label_pad_token_id=self.tokenizer.pad_token_id,
            pad_to_multiple_of=8 if self.fp16 else None,
        )
              
        # Define loss function
        weight = torch.ones(self.tokenizer.__len__())
        self.loss = nn.NLLLoss(weight=weight, ignore_index=self.tokenizer.pad_token_id)
        
        
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
        
        
    def training_step(self, batch, batch_idx):
        input_variables = getattr(batch, 'input_ids')
        input_lengths = getattr(batch, 'input_lens')
        target_variables = getattr(batch, 'labels')
        criterion = self.loss
        # Forward propagation
        decoder_outputs, decoder_hidden, other = self.forward(
            input_variables, 
            input_lengths, 
            target_variables,
            teacher_forcing_ratio=self.teacher_forcing_ratio,
        )
        # Get loss
        for step, step_output in enumerate(decoder_outputs):
            batch_size = target_variables.size(0)
            loss = criterion(step_output.contiguous().view(batch_size, -1), target_variables[:, step + 1])
            
        return loss
    
    
    @torch.no_grad()
    def predict_sequence(self, sent: str):
        # Encode
        input_ids = self.tokenizer.encode(
            sent, 
            max_length=self.max_len, 
            truncation=True, 
            return_tensors='pt',
        )
        input_lens = torch.tensor([input_ids.size(1)])
        
        # Inference
        _, _, other = self.forward(input_ids, input_lens)
        length = other['length'][0]
        tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        tgt_seq = self.tokenizer.batch_decode(tgt_id_seq, skip_special_tokens=True)[0]
        return tgt_seq, len(tgt_id_seq)
        
    
    # def predict_batch(self, batch, batch_idx):
    #     input_variables = getattr(batch, 'input_ids')
    #     input_lengths = getattr(batch, 'input_lens')
    #     target_variables = getattr(batch, 'labels')
        
    #     # Forward propagation
    #     _, _, other = self.model(
    #         input_variables, 
    #         input_lengths, 
    #         target_variables,
    #         teacher_forcing_ratio=self.teacher_forcing_ratio,
    #     )
    #     lengths = other['length']
        
    #     # Decode
    #     tgt_id_seqs = [[other['sequence'][di][0].data[0] for di in range(len)] for len in lengths]
    #     # tgt_seqs = self.tokenizer.batch_decode(tgt_id_seqs, skip_special_tokens=True)
    #     return {
    #         'input_ids': input_variables,
    #         'pred_ids': tgt_id_seqs,   
    #         # 'pred_seqs': tgt_seqs,
    #         'labels': target_variables,
    #     }
    
    
    @torch.no_grad()
    def _eval_step(
        self, 
        batch: Sequence[torch.Tensor],
        batch_idx: int, 
        split: str,
    ):
        loss = self.training_step(batch, batch_idx)
        self.log(f'{split}_loss', loss)
        return loss
        
    
    def validation_step(self, batch, batch_idx):
        loss = self._eval_step(batch, batch_idx, 'val')
        return loss
    
    
    def test_step(self, batch, batch_idx):
        loss = self._eval_step(batch, batch_idx, 'test')
        return loss
    
    
    def configure_optimizers(self):
        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        effective_batch_size = self.n_gpus * self.train_batch_size
        scaled_lr = self.base_lr * math.sqrt(effective_batch_size) if self.base_lr else None
        print('Effective learning rate: {}'.format(scaled_lr))
        optimizer = Adam(self.model.parameters(), lr=scaled_lr)
        return optimizer
    
    
    def set_optimizer_state(self, state_dict):
        """
        Sets optimizer state dictionary if loading from a checkpoint
        Args:
            state_dict (dict): Optimizer state dictionary
        """
        self.optimizer_state_dict = state_dict
        self.log('Intializing optimizers with existing state dictionary')
    
    
    def get_split_dataloader(self, split: str):
        # Load split
        if split == 'val':
            split = 'validation'
            
        dataset = Seq2SeqDataset(
            data_file=self.data_files[split],
            tokenizer=self.tokenizer,
            max_len=self.max_len,
            src_field_name=self.src_field_name,
            tgt_field_name=self.tgt_field_name,
        )
        print('Created {} {} of {:,} size'.format(
            split, dataset.__class__.__name__, len(dataset)))
        
        if self.n_gpus == 1:
            sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        print('Created {}'.format(sampler.__class__.__name__))
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=self.train_batch_size 
            if split == 'train' else self.eval_batch_size,
            collate_fn=self.data_collator,
            pin_memory=True,
        )
        print('Created {}'.format(dataloader.__class__.__name__))
        return dataloader
    

    def train_dataloader(self):
        return self.get_split_dataloader('train')
    
    def val_dataloader(self):
        return self.get_split_dataloader('val')
    
    def test_dataloader(self):
        return self.get_split_dataloader('test')
    
    def eval_epoch_end(self, outputs: List[dict], split: str):
        total_loss = sum(outputs)
        print('Average {} loss: {}'.format(split, total_loss / len(outputs)))
        
    def validation_epoch_end(self, outputs: List[dict]):
        self.eval_epoch_end(outputs, 'val')
        
    def test_epoch_end(self, outputs: List[dict]):
        self.eval_epoch_end(outputs, 'test')
        
        
    #     # Convert the predict and label ids to lists of tokens
    #     df = pd.DataFrame(outputs).groupby('a').agg(list)
        
    #     for ins in outputs:
    #         hypothesis = [str(id) for id in ins['tgt_ids']]
    #         references = [str(id) for id in ins['labels']]

    #     # Calculate the BLEU score
    #     bleu_score = sentence_bleu(label_tokens, predict_tokens)
    #     self.log('bleu_score', bleu_score)
    #     print('BLEU score: {}'.format(bleu_score))

        # # Calculate the ROUGE-L score
        # rouge_l_score = rouge_l(label_tokens, predict_tokens)
        # self.log('rouge_l_score', rouge_l_score)
        # print('BLEU score: {}, ROUGE-L: {}'.format(bleu_score, rouge_l_score))




if __name__ == '__main__':
    import time
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tok_path', type=str, default='../DG_ckpt/bart')
    parser.add_argument('--data_files', type=dict, default=DEFAULT_DATA)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--bidirectional', type=bool, default=True)
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5)
    parser.add_argument('--input_dropout', type=float, default=0)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--rnn_cell', type=str, default='gru')
    parser.add_argument('--attention', type=bool, default=False)
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--src_field_name', type=str, default='src')
    parser.add_argument('--tgt_field_name', type=str, default='tgt')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--no_early_stopping', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    
    args = parser.parse_args()
    
    start = time.time()
    
    # Reproductibility
    set_seed(args.seed, gpu=(args.n_gpus > 0))
    
    # Make the model output / checkpoint directory
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    model = Seq2SeqSystem(args)
    
    # Load checkpoint if it exists
    if args.do_train:
        print('!! TRAINING FROM SCRATCH !!')
    elif os.path.exists(args.ckpt_dir):
        checkpoints = glob.glob(os.path.join(args.ckpt_dir, '*.ckpt'))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
            print('!! RESUMING FROM CHECKPOINT: {} !!'.format(latest_checkpoint))
            # Custom loading because lightning seems to be broken.
            checkpoint = torch.load(
                latest_checkpoint,
                map_location=lambda storage, loc: storage)
            state_dict = checkpoint['state_dict']
            # Correct some params
            all_keys = list(state_dict.keys())
            for k in tqdm(all_keys):
                if k.endswith('residual_alpha'):
                    state_dict[k.replace('residual_alpha', 'r_alpha')] = state_dict.pop(k)
            print('Corrected state dicts')
            print('Loading parameters from:\n{}'.format(list(state_dict.keys())))
            model.load_state_dict(state_dict, strict=True)
            # Optimizer states
            # model.set_optimizer_state(checkpoint['optimizer_states'][0])
            # give model a chance to load something
            model.on_load_checkpoint(checkpoint)
    else:
        print('!! TRAINING FROM SCRATCH !!')
    
    # Most basic trainer, uses good defaults
    model_checkpoint = ModelCheckpoint(
        monitor='val_loss',
        dirpath=args.ckpt_dir,
        save_top_k=5,
        every_n_epochs=1,
    )
    
    # Earlystop default is overridden by this
    early_stop_callback = False if args.no_early_stopping else \
        EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=4,
            verbose=True,
            mode='min',
        )
    
    trainer = Trainer(
        limit_train_batches=100, 
        min_epochs=min(5, args.max_epochs),
        max_epochs=args.max_epochs,
        gpus=args.n_gpus,
        callbacks=[model_checkpoint, early_stop_callback],
    )
    
    print('{} - Created {} object'.format(
        time.time() - start, trainer.__class__.__name__
    ))
    
    # Fitting the model
    if args.do_train:
        trainer.fit(model)
        print('{} - Fitted model {}'.format(
            time.time() - start, os.path.basename(args.ckpt_dir)
        ))
    
    # Testing the model
    if args.do_test:
        test_start = time.time()
        trainer.test(model)
        print('{} - Tested model {}'.format(
            time.time() - test_start, os.path.basename(args.ckpt_dir)
        ))
        
    while True:
        pred, pred_len = model.predict_sequence('I am a student, what do you do?')
        print(f"prediction ({pred_len}): {pred}")
        break
    