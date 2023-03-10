
import language_tool_python
from sentence_transformers import SentenceTransformer, util

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
        self.lang_tool = language_tool_python.LanguageTool('en-US')

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