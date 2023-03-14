
import torch
import random
import numpy as np
import language_tool_python
from sentence_transformers import SentenceTransformer, util
import re
import spacy
nlp = spacy.load("en_core_web_sm")
from nltk.tokenize import word_tokenize

MAX_PER_DICT = {
    'l2': 40,
    'linf': 0.03
}

SAMPLE_NOISE = "tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav"

ENGLISH_FILTER_WORDS = [
    'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
    'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
    'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
    'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
    'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
    "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
    'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
    'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
    'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
    'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
    'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
    "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
    'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
    'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
    'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
    'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
    'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
    'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
    'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
    'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
    'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
    'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
    'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
    "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
    'your', 'yours', 'yourself', 'yourselves', 'have', 'be'
]

DEFAULT_TEMPLATES = [
    '( ROOT ( S ( NP ) ( VP ) ( . ) ) ) EOP',
    '( ROOT ( S ( VP ) ( . ) ) ) EOP',
    '( ROOT ( NP ( NP ) ( . ) ) ) EOP',
    '( ROOT ( FRAG ( SBAR ) ( . ) ) ) EOP',
    '( ROOT ( S ( S ) ( , ) ( CC ) ( S ) ( . ) ) ) EOP',
    '( ROOT ( S ( LST ) ( VP ) ( . ) ) ) EOP',
    '( ROOT ( SBARQ ( WHADVP ) ( SQ ) ( . ) ) ) EOP',
    '( ROOT ( S ( PP ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP',
    '( ROOT ( S ( ADVP ) ( NP ) ( VP ) ( . ) ) ) EOP',
    '( ROOT ( S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP'
]


class GrammarChecker:
    def __init__(self):
        # self.lang_tool = language_tool_python.LanguageTool('en-US')
        self.lang_tool = language_tool_python.LanguageToolPublicAPI('es')

    def check(self, sentence):
        '''
        :param sentence:  a string
        :return:
        '''
        matches = self.lang_tool.check(sentence)
        return len(matches)


class SentenceEncoder:
    def __init__(self, device='cuda'):
        '''
        different version of Universal Sentence Encoder
        https://pypi.org/project/sentence-transformers/
        '''
        self.model = SentenceTransformer('paraphrase-distilroberta-base-v1', device)

    def encode(self, sentences):
        '''
        can modify this code to allow batch sentences input
        :param sentence: a String
        :return:
        '''
        if isinstance(sentences, str):
            sentences = [sentences]
        return self.model.encode(sentences, convert_to_tensor=True)

    def get_sim(self, sentence1: str, sentence2: str):
        '''
        can modify this code to allow batch sentences input
        :param sentence1: a String
        :param sentence2: a String
        :return:
        '''
        embeddings = self.model.encode([sentence1, sentence2], convert_to_tensor=True)
        cos_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return cos_sim.item()

    # find adversarial sample in advs which matches ori best
    def find_best_sim(self, ori, advs, find_min=False):
        ori_embedding = self.model.encode(ori, convert_to_tensor=True)
        adv_embeddings = self.model.encode(advs, convert_to_tensor=True)
        best_adv = None
        best_index = None
        best_sim = 10 if find_min else -10
        for i, adv_embedding in enumerate(adv_embeddings):
            sim = util.pytorch_cos_sim(ori_embedding, adv_embedding).item()
            if find_min:
                if sim < best_sim:
                    best_sim = sim
                    best_adv = advs[i]
                    best_index = i
            else:
                if sim > best_sim:
                    best_sim = sim
                    best_adv = advs[i]
                    best_index = i

        return best_adv, best_index, best_sim
    
    
    
def set_seed(seed: int, gpu: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu:
        torch.cuda.manual_seed_all(seed)
    print('Set seeds to {}'.format(seed))
    
    
    

def output_score1(input_sentence: str):
    # Parse the input sentence with a NLP tool
    doc = nlp(input_sentence)

    # Calculate a base score based on the number of words
    num_words = len(doc)
    base_score = num_words / 30  # Assumes that an average sentence length is around 30 words

    # Adjust the score based on the linguistic properties of the input sentence
    for token in doc:
        # Increase score for longer words
        if len(token.text) > 6:
            base_score += 0.1

        # Increase score for complex syntactic structures
        if token.dep_ in ["ccomp", "xcomp", "advcl", "relcl", "acl"]:
            base_score += 0.2

        # Increase score for ambiguous or multi-interpretation sentences
        if token.pos_ in ["VERB", "NOUN", "ADJ"]:
            for child in token.children:
                if child.pos_ == "NOUN" or child.pos_ == "ADJ":
                    base_score += 0.1
                if child.dep_ == "advmod":
                    base_score += 0.1

        # Increase score for dense information sentences
        if token.pos_ == "NOUN" or token.pos_ == "VERB":
            if len(token.text) > 4:
                base_score += 0.1
                
    return base_score


def output_score2(input_sentence: str):
    score = 0
    
    # Check for open-ended questions
    if input_sentence.endswith("?"):
        score += 2
        
    # Check for ambiguous or vague language
    if len(input_sentence.split()) > 10:
        score += 1
        
    # Check for multi-part questions
    num_commas = len(re.findall(",", input_sentence))
    num_and = len(re.findall(" and ", input_sentence))
    if num_commas > 0 or num_and > 0:
        score += num_commas + num_and
        
    # Check for unfamiliar or complex terminology
    doc = nlp(input_sentence)
    tech_terms = ["epigenetics", "gene expression", "artificial intelligence", "machine learning"]
    num_tech_terms = sum([1 for token in doc if token.text.lower() in tech_terms])
    score += num_tech_terms
    
    # Check for abstract or philosophical concepts
    abs_terms = ["consciousness", "self", "reality", "ontology", "phenomenology"]
    num_abs_terms = sum([1 for token in doc if token.text.lower() in abs_terms])
    score += num_abs_terms
    
    return score


def output_score3(input_sentence: str):
    score = 0
    
    # Check for open-ended questions
    if input_sentence.endswith("?"):
        score += 2
        
    # Check for ambiguous or vague language
    if len(input_sentence.split()) > 10:
        score += 1
        
    # Check for multi-part questions
    num_commas = len(re.findall(",", input_sentence))
    num_and = len(re.findall(" and ", input_sentence))
    if num_commas > 0 or num_and > 0:
        score += num_commas + num_and
        
    # Check for unfamiliar or complex terminology
    doc = nlp(input_sentence)
    for token in doc:
        if token.lemma_ in ["epigenetic", "gene expression"]:
            score += 3
        
    # Check for abstract or philosophical concepts
    if "nature of consciousness" in input_sentence or "self and the world" in input_sentence:
        score += 4
        
    # Tokenize input sentence
    tokens = word_tokenize(input_sentence)
    
    # Count number of tokens
    num_tokens = len(tokens)
    
    # Calculate average length of tokens
    avg_token_length = sum(len(token) for token in tokens) / num_tokens
    
    # Adjust score based on token length
    if avg_token_length > 8:
        score += 1
    elif avg_token_length > 10:
        score += 2
    elif avg_token_length > 12:
        score += 3
    
    return score



def open_ended_question_score(input_sentence):
    if input_sentence.endswith("?"):
        return 1.0
    else:
        return 0.0

def ambiguous_language_score(input_sentence):
    if len(input_sentence.split()) > 10:
        return 0.5
    else:
        return 0.0

def multi_part_question_score(input_sentence):
    num_commas = len(re.findall(",", input_sentence))
    num_and = len(re.findall(" and ", input_sentence))
    num_parts = num_commas + num_and
    return num_parts * 0.5

def unfamiliar_terminology_score(input_sentence):
    doc = nlp(input_sentence)
    unfamiliar_terms = 0
    for token in doc:
        if token.lemma_ in ["epigenetic", "gene expression", "quantum mechanics", "neuroplasticity", "postmodernism"]:
            unfamiliar_terms += 1
    return unfamiliar_terms * 0.5

def abstract_concepts_score(input_sentence):
    if "nature of consciousness" in input_sentence or "self and the world" in input_sentence or "meaning of life" in input_sentence:
        return 1.0
    else:
        return 0.0
    
def token_length_score(input_sentence):
    tokens = word_tokenize(input_sentence)
    num_tokens = len(tokens)
    avg_token_length = sum(len(token) for token in tokens) / num_tokens
    if avg_token_length > 12:
        return 1.0
    elif avg_token_length > 10:
        return 0.5
    else:
        return 0.0
    
def input_length_score(input_sentence):
    if len(input_sentence) > 200:
        return 1.0
    elif len(input_sentence) > 100:
        return 0.5
    else:
        return 0.0

def output_score4(input_sentence):
    score = open_ended_question_score(input_sentence)
    score += ambiguous_language_score(input_sentence)
    score += multi_part_question_score(input_sentence)
    score += unfamiliar_terminology_score(input_sentence)
    score += abstract_concepts_score(input_sentence)
    score += token_length_score(input_sentence)
    score += input_length_score(input_sentence)
    return score