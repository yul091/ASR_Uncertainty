import copy
import logging
from typing import Dict
import torch
import torch.nn as nn
from alpaca.uncertainty_estimator.masks import build_mask
from transformers import ElectraForSequenceClassification
from transformers.activations import get_activation

log = logging.getLogger(__name__)


class DropoutMC(torch.nn.Module):
    def __init__(self, p: float, activate=False):
        super().__init__()
        self.activate = activate
        self.p = p
        self.p_init = p

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.dropout(x, self.p, training=self.training or self.activate)


class LockedDropoutMC(DropoutMC):
    """
    Implementation of locked (or variational) dropout. Randomly drops out entire parameters in embedding space.
    """

    def __init__(self, p: float, activate: bool = False, batch_first: bool = True):
        super().__init__(p, activate)
        self.batch_first = batch_first

    def forward(self, x):
        if self.training:
            self.activate = True
        # if not self.training or not self.p:
        if not self.activate or not self.p:
            return x

        if not self.batch_first:
            m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.p)
        else:
            m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p)

        mask = torch.autograd.Variable(m, requires_grad=False) / (1 - self.p)
        mask = mask.expand_as(x)
        return mask * x


class WordDropoutMC(DropoutMC):
    """
    Implementation of word dropout. Randomly drops out entire words (or characters) in embedding space.
    """

    def forward(self, x):
        if self.training:
            self.activate = True

        # if not self.training or not self.p:
        if not self.activate or not self.p:
            return x

        m = x.data.new(x.size(0), x.size(1), 1).bernoulli_(1 - self.p)

        mask = torch.autograd.Variable(m, requires_grad=False)
        return mask * x


MC_DROPOUT_SUBSTITUTES = {
    "Dropout": DropoutMC,
    "LockedDropout": LockedDropoutMC,
    "WordDropout": WordDropoutMC,
}



class DropoutDPP(DropoutMC):
    dropout_id = -1

    def __init__(
        self,
        p: float,
        activate=False,
        mask_name="dpp",
        max_n=100,
        max_frac=0.4,
        coef=1.0,
    ):
        super().__init__(p=p, activate=activate)

        self.mask = build_mask(mask_name)
        self.reset_mask = False
        self.max_n = max_n
        self.max_frac = max_frac
        self.coef = coef

        self.curr_dropout_id = DropoutDPP.update()
        log.debug(f"Dropout id: {self.curr_dropout_id}")

    @classmethod
    def update(cls):
        cls.dropout_id += 1
        return cls.dropout_id

    def calc_mask(self, x: torch.Tensor):
        return self.mask(x, dropout_rate=self.p, layer_num=self.curr_dropout_id).float()

    def get_mask(self, x: torch.Tensor):
        return self.mask(x, dropout_rate=self.p, layer_num=self.curr_dropout_id).float()

    def calc_non_zero_neurons(self, sum_mask):
        frac_nonzero = (sum_mask != 0).sum(axis=-1).item() / sum_mask.shape[-1]
        return frac_nonzero

    def forward(self, x: torch.Tensor):
        if self.training:
            return torch.nn.functional.dropout(x, self.p, training=True)
        else:
            if not self.activate:
                return x

            sum_mask = self.get_mask(x)

            norm = 1.0
            i = 1
            frac_nonzero = self.calc_non_zero_neurons(sum_mask)
            # print('==========Non zero neurons:', frac_nonzero, 'iter:', i, 'id:', self.curr_dropout_id, '******************')
            # while i < 30:
            while i < self.max_n and frac_nonzero < self.max_frac:
                # while frac_nonzero < self.max_frac:
                mask = self.get_mask(x)

                # sum_mask = self.coef * sum_mask + mask
                sum_mask += mask
                i += 1
                # norm = self.coef * norm + 1

                frac_nonzero = self.calc_non_zero_neurons(sum_mask)
                # print('==========Non zero neurons:', frac_nonzero, 'iter:', i, '******************')

            # res = x * sum_mask / norm
            print("Number of masks:", i)
            res = x * sum_mask / i
            return res


class ElectraClassificationHeadCustom(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, other):
        super().__init__()
        self.dropout1 = other.dropout
        self.dense = other.dense
        self.dropout2 = copy.deepcopy(other.dropout)
        self.out_proj = other.out_proj

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout1(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout2(x)
        x = self.out_proj(x)
        return x


def convert_to_mc_dropout(model: torch.nn.Module, substitution_dict: Dict[str, torch.nn.Module] = None):
    for i, layer in enumerate(list(model.children())):
        proba_field_name = "dropout_rate" if "flair" in str(type(layer)) else "p"
        module_name = list(model._modules.items())[i][0]
        layer_name = layer._get_name()
        if layer_name in substitution_dict.keys():
            model._modules[module_name] = substitution_dict[layer_name](
                p=getattr(layer, proba_field_name), activate=False
            )
        else:
            convert_to_mc_dropout(model=layer, substitution_dict=substitution_dict)


def activate_mc_dropout(model: torch.nn.Module, activate: bool, random: float = 0.0, verbose: bool = False):
    for layer in model.children():
        if isinstance(layer, DropoutMC):
            if verbose:
                print(layer)
                print(f"Current DO state: {layer.activate}")
                print(f"Switching state to: {activate}")
            layer.activate = activate
            if activate and random:
                layer.p = random
            if not activate:
                layer.p = layer.p_init
        else:
            activate_mc_dropout(model=layer, activate=activate, random=random, verbose=verbose)


def get_last_dropout(model):
    if isinstance(model, ElectraForSequenceClassification):
        if isinstance(model.classifier, ElectraClassificationHeadCustom):
            return model.classifier.dropout2
        else:
            return model.classifier.dropout
    else:
        return model.dropout


def set_last_dropout(model, dropout):
    if isinstance(model, ElectraForSequenceClassification):
        if isinstance(model.classifier, ElectraClassificationHeadCustom):
            model.classifier.dropout2 = dropout
        else:
            model.classifier.dropout
    else:
        model.dropout = dropout


def convert_dropouts(model, ue_args):
    if ue_args.dropout_type == "MC":
        dropout_ctor = lambda p, activate: DropoutMC(p=ue_args.inference_prob, activate=False)
    elif ue_args.dropout_type == "DPP":
        def dropout_ctor(p, activate):
            return DropoutDPP(
                p=p,
                activate=activate,
                max_n=ue_args.dropout.max_n,
                max_frac=ue_args.dropout.max_frac,
                mask_name=ue_args.dropout.mask_name,
            )
    else:
        raise ValueError(f"Wrong dropout type: {ue_args.dropout_type}")

    if ue_args.dropout_subs == "last":
        set_last_dropout(model, dropout_ctor(p=ue_args.inference_prob, activate=False))

    elif ue_args.dropout_subs == "all":
        convert_to_mc_dropout(model, {'Dropout': dropout_ctor})
        # convert_to_mc_dropout(model.electra.encoder, {"Dropout": dropout_ctor})
    else:
        raise ValueError(f"Wrong ue args {ue_args.dropout_subs}")
    

def calculate_dropouts(model):
    res = 0
    for i, layer in enumerate(list(model.children())):
        module_name = list(model._modules.items())[i][0]
        layer_name = layer._get_name()
        if layer_name == "Dropout":
            res += 1
        else:
            res += calculate_dropouts(model=layer)

    return res


def freeze_all_dpp_dropouts(model, freeze):
    for layer in model.children():
        if isinstance(layer, DropoutDPP):
            if freeze:
                layer.mask.freeze(dry_run=True)
            else:
                layer.mask.unfreeze(dry_run=True)
        else:
            freeze_all_dpp_dropouts(model=layer, freeze=freeze)