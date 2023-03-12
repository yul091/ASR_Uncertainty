import sys
sys.dont_write_bytecode = True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # avoid tensorflow warnings
import time
import argparse
import random
import numpy as np
from tqdm import tqdm
from typing import List
import torch
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)
from datasets import load_dataset, Dataset
import evaluate
from DialogueAPI import dialogue
from attackers import WordAttacker, StructureAttacker
from DGdataset import DGDataset

DATA2NAME = {
    "blended_skill_talk": "BST",
    "conv_ai_2": "ConvAI2",
    "empathetic_dialogues": "ED",
    "AlekseyKorshuk/persona-chat": "PC",
}

class DGAttackEval(DGDataset):
    def __init__(
        self, 
        args: argparse.Namespace = None, 
        tokenizer: AutoTokenizer = None, 
        model: AutoModelForSeq2SeqLM = None, 
        attacker: WordAttacker = None, 
        device: torch.device('cpu') = None, 
        task: str = 'seq2seq', 
        bleu: evaluate.load("bleu") = None, 
        rouge: evaluate.load("rouge") = None,
        meteor: evaluate.load("meteor") = None,
        pred_only: bool = False,
    ):
        super(DGAttackEval, self).__init__(
            dataset=args.dataset,
            task=task,
            tokenizer=tokenizer,
            max_source_length=args.max_len,
            max_target_length=args.max_len,
            padding=None,
            ignore_pad_token_for_loss=True,
            preprocessing_num_workers=None,
            overwrite_cache=True,
        )

        self.args = args
        self.model = model
        self.attacker = attacker
        self.device = device
        self.sp_token = attacker.sp_token

        self.num_beams = args.num_beams 
        self.num_beam_groups = args.num_beam_groups
        self.max_num_samples = args.max_num_samples

        self.bleu = bleu
        self.rouge = rouge
        self.meteor = meteor
        self.pred_only = pred_only

        self.ori_lens, self.adv_lens = [], []
        self.ori_bleus, self.adv_bleus = [], []
        self.ori_rouges, self.adv_rouges = [], []
        self.ori_meteors, self.adv_meteors = [], []
        self.ori_time, self.adv_time = [], []
        self.cos_sims = []
        self.att_success = 0
        self.total_pairs = 0
        
        att_method = args.attack_strategy
        self.out_dir = args.out_dir
        self.model_n = args.model_name_or_path.split("/")[-1]
        dataset_n = DATA2NAME.get(args.dataset, args.dataset.split("/")[-1])
        combined = "combined" if args.use_combined_loss and att_method == 'structure' else "single"
        max_per = args.max_per
        fitness = args.fitness
        select_beams = args.select_beams
        max_n_samples = args.max_num_samples
        
        if self.pred_only:
            log_path = f"{self.out_dir}/{self.model_n}_{dataset_n}_{max_n_samples}.txt"
            self.write_file = open(log_path, "w")
            self.res_path = f"{self.out_dir}/{self.model_n}_{dataset_n}_{max_n_samples}.res"
            self.pred_res = []
        else:
            log_path = f"{self.out_dir}/{att_method}_{combined}_{max_per}_{fitness}_{select_beams}_{self.model_n}_{dataset_n}_{max_n_samples}.txt"
            self.write_file = open(log_path, "w")
            self.res_path = f"{self.out_dir}/{att_method}_{max_per}_{fitness}_{select_beams}_{self.model_n}_{dataset_n}_{max_n_samples}.res"
        
        
        
    def log_and_save(self, display: str):
        print(display)
        self.write_file.write(display + "\n")
        

    def get_prediction(self, text: str):
        if self.task == 'seq2seq':
            effective_text = text 
        else:
            effective_text = text + self.tokenizer.eos_token

        inputs = self.tokenizer(
            effective_text,  
            return_tensors="pt",
            truncation=True,
            max_length=self.max_source_length-1,
        )
        input_ids = inputs.input_ids.to(self.device)
        t1 = time.time()
        with torch.no_grad():
            outputs = dialogue(
                self.model, 
                input_ids,
                early_stopping=False, 
                num_beams=self.num_beams,
                num_beam_groups=self.num_beam_groups, 
                use_cache=True,
                max_length=self.max_target_length,
            )
        if self.task == 'seq2seq':
            output = self.tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)[0]
        else:
            output = self.tokenizer.batch_decode(
                outputs['sequences'][:, input_ids.shape[-1]:], 
                skip_special_tokens=True,
            )[0]
        t2 = time.time()
        return output.strip(), t2 - t1


    def eval_metrics(self, output: str, guided_messages: List[str]):
        if not output:
            return

        bleu_res = self.bleu.compute(
            predictions=[output], 
            references=[guided_messages],
            smooth=True,
        )
        rouge_res = self.rouge.compute(
            predictions=[output],
            references=[guided_messages],
        )
        meteor_res = self.meteor.compute(
            predictions=[output],
            references=[guided_messages],
        )
        pred_len = bleu_res['translation_length']
        return bleu_res, rouge_res, meteor_res, pred_len
        
        
    def generation_step(self, instance: dict):
        # Set up
        num_entries, total_entries, context, prev_utt_pc = self.prepare_context(instance)
        for entry_idx in range(num_entries):
            free_message, guided_message, original_context, references = self.prepare_entry(
                instance, 
                entry_idx, 
                context, 
                prev_utt_pc,
                total_entries,
            )
            if guided_message is None:
                continue
            
            prev_utt_pc += [
                free_message,
                guided_message,
            ]

            self.log_and_save(f"\nDialogue history: {original_context}")
            self.log_and_save("U--{} \n(Ref: ['{}', ...])".format(free_message, references[-1]))
            # Original generation
            text = original_context + self.sp_token + free_message
            output, time_gap = self.get_prediction(text)
            self.log_and_save(f"G--{output}")
            
            if not output:
                continue
            bleu_res, rouge_res, meteor_res, pred_len = self.eval_metrics(output, references)
            if self.pred_only:
                self.pred_res.append((
                    self.model_n,
                    text, 
                    output,
                    references,
                    pred_len, 
                    bleu_res['bleu'], 
                    rouge_res['rougeL'], 
                    meteor_res['meteor'],
                    time_gap, 
                ))
            
            self.log_and_save("(length: {}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f})".format(
                pred_len, time_gap, bleu_res['bleu'], rouge_res['rougeL'], meteor_res['meteor'],
            ))
            self.ori_lens.append(pred_len)
            self.ori_bleus.append(bleu_res['bleu'])
            self.ori_rouges.append(rouge_res['rougeL'])
            self.ori_meteors.append(meteor_res['meteor'])
            self.ori_time.append(time_gap)
            
            # Attack
            if not self.pred_only:
                success, new_text, new_len = self.attacker.run_attack(text, guided_message)
                new_free_message = new_text.split(self.sp_token)[1].strip()
                cos_sim = self.attacker.sent_encoder.get_sim(new_free_message, free_message)
                output, time_gap = self.get_prediction(new_text)
                if not output:
                    continue

                self.log_and_save("U'--{} (cosine: {:.3f})".format(new_free_message, cos_sim))
                self.log_and_save(f"G'--{output}")
                adv_bleu_res, adv_rouge_res, adv_meteor_res, adv_pred_len = self.eval_metrics(output, references)
                
                # ASR
                success = (
                    (bleu_res['bleu'] > adv_bleu_res['bleu']) or 
                    (rouge_res['rougeL'] > adv_rouge_res['rougeL']) or 
                    (meteor_res['meteor'] > adv_meteor_res['meteor'])
                    ) and cos_sim > 0.01
                if success:
                    self.att_success += 1
                else:
                    self.log_and_save("Attack failed!")

                self.log_and_save("(length: {}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f})".format(
                    adv_pred_len, time_gap, adv_bleu_res['bleu'], adv_rouge_res['rougeL'], adv_meteor_res['meteor'],
                ))
                self.adv_lens.append(adv_pred_len)
                self.adv_bleus.append(adv_bleu_res['bleu'])
                self.adv_rouges.append(adv_rouge_res['rougeL'])
                self.adv_meteors.append(adv_meteor_res['meteor'])
                self.adv_time.append(time_gap)
                self.cos_sims.append(cos_sim)
                self.total_pairs += 1


    def generation(self, test_dataset: Dataset):
        if self.dataset == "empathetic_dialogues":
            test_dataset = self.group_ED(test_dataset)

        # Sample test dataset
        if self.max_num_samples > 0:
            print(f"Sampling {self.max_num_samples} samples from test dataset...")
            ids = random.sample(range(len(test_dataset)), self.max_num_samples)
            test_dataset = test_dataset.select(ids)
            
        print(f"Test dataset: {test_dataset}")
        for i, instance in tqdm(enumerate(test_dataset)):
            self.generation_step(instance)

        Ori_len = np.mean(self.ori_lens)
        Ori_bleu = np.mean(self.ori_bleus)
        Ori_rouge = np.mean(self.ori_rouges)
        Ori_meteor = np.mean(self.ori_meteors)
        Ori_t = np.mean(self.ori_time)
        
        self.log_and_save("\nOriginal output length: {:.3f}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f}".format(
            Ori_len, Ori_t, Ori_bleu, Ori_rouge, Ori_meteor,
        ))
        
        if self.pred_only:
            torch.save(self.pred_res, self.res_path)
        else:
            Adv_len = np.mean(self.adv_lens)
            Adv_bleu = np.mean(self.adv_bleus)
            Adv_rouge = np.mean(self.adv_rouges)
            Adv_meteor = np.mean(self.adv_meteors)
            Cos_sims = np.mean(self.cos_sims)
            Adv_t = np.mean(self.adv_time)
        
            self.log_and_save("Perturbed [cosine: {:.3f}] output length: {:.3f}, latency: {:.3f}, BLEU: {:.3f}, ROUGE: {:.3f}, METEOR: {:.3f}".format(
                Cos_sims, Adv_len, Adv_t, Adv_bleu, Adv_rouge, Adv_meteor,
            ))
            self.log_and_save("Attack success rate: {:.2f}%".format(100*self.att_success/self.total_pairs))
            # Save adv samples
            torch.save(self.attacker.adv_his, self.res_path)


def main(args: argparse.Namespace):
    random.seed(args.seed)
    model_name_or_path = args.model_name_or_path
    dataset = args.dataset
    max_len = args.max_len
    max_per = args.max_per
    num_beams = args.num_beams
    select_beams = args.select_beams
    fitness = args.fitness
    num_beam_groups = args.num_beam_groups
    att_method = args.attack_strategy
    cls_weight = args.cls_weight
    eos_weight = args.eos_weight
    pred_only = args.pred_only
    use_combined_loss = args.use_combined_loss
    out_dir = args.out_dir

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    config = AutoConfig.from_pretrained(model_name_or_path, num_beams=num_beams, num_beam_groups=num_beam_groups)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if 'gpt' in model_name_or_path.lower():
        task = 'clm'
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config)
        if 'results' not in model_name_or_path.lower():
            tokenizer.add_special_tokens({'pad_token': '<PAD>'})
            tokenizer.add_special_tokens({'mask_token': '<MASK>'})
            model.resize_token_embeddings(len(tokenizer))
    else:
        task = 'seq2seq'
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config)

    # Load dataset
    all_datasets = load_dataset(dataset)
    if dataset == "conv_ai_2":
        test_dataset = all_datasets['train']
    elif dataset == "AlekseyKorshuk/persona-chat":
        test_dataset = all_datasets['validation']
    else:
        test_dataset = all_datasets['test']

    # Define attack method
    if att_method.lower() == 'word':
        attacker = WordAttacker(
            device=device,
            tokenizer=tokenizer,
            model=model,
            max_len=max_len,
            max_per=max_per,
            task=task,
            fitness=fitness,
            select_beams=select_beams,
        )
    elif att_method.lower() == 'structure':
        attacker = StructureAttacker(
            device=device,
            tokenizer=tokenizer,
            model=model,
            max_len=max_len,
            max_per=max_per,
            task=task,
            fitness=fitness,
            select_beams=select_beams,
            eos_weight=eos_weight,
            cls_weight=cls_weight,
            use_combined_loss=use_combined_loss,
        )
    else:
        raise ValueError("Invalid attack strategy!")

    # Load evaluation metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    # Define DG attack
    dg = DGAttackEval(
        args=args,
        tokenizer=tokenizer,
        model=model,
        attacker=attacker,
        device=device,
        task=task,
        bleu=bleu,
        rouge=rouge,
        meteor=meteor,
        pred_only=pred_only,
    )
    dg.generation(test_dataset)



if __name__ == "__main__":
    import ssl
    import argparse
    import nltk
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')
    ssl._create_default_https_context = ssl._create_unverified_context

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_num_samples", type=int, default=-1, help="Number of samples to attack")
    parser.add_argument("--max_per", type=int, default=5, help="Number of perturbation iterations per sample")
    parser.add_argument("--max_len", type=int, default=1024, help="Maximum length of generated sequence")
    parser.add_argument("--num_beams", type=int, default=2, help="Number of beams")
    parser.add_argument("--select_beams", type=int, default=1, help="Number of sentence beams to keep for each attack iteration")
    parser.add_argument("--num_beam_groups", type=int, default=1, help="Number of beam groups")
    parser.add_argument("--fitness", type=str, default="length", 
                        choices=["length", "performance", "random", "combined"],
                        help="Fitness function")
    parser.add_argument("--model_name_or_path", "-m", type=str, default="results/bart", help="Path to model")
    parser.add_argument("--dataset", "-d", type=str, default="blended_skill_talk", 
                        choices=[
                            "blended_skill_talk",
                            "conv_ai_2",
                            "empathetic_dialogues",
                            "AlekseyKorshuk/persona-chat",
                        ], 
                        help="Dataset to attack")
    parser.add_argument("--out_dir", type=str,
                        default="results/logging",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=2019, help="Random seed")
    parser.add_argument("--pred_only", action="store_true", help="Whether to attack or just predict")
    parser.add_argument("--eos_weight", type=float, default=0.8, help="Weight for EOS gradient")
    parser.add_argument("--cls_weight", type=float, default=0.2, help="Weight for classification gradient")
    parser.add_argument("--use_combined_loss", action="store_true", help="Use combined loss")
    parser.add_argument("--attack_strategy", "-a", type=str, default='word', 
                        choices=['structure', 'word'], 
                        help="Attack strategy")
    args = parser.parse_args()
    main(args)
