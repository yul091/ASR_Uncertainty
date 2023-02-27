import torch
# from typing import List


def data_uncertainty(preds: torch.Tensor, ue: str = "vanilla"):
    """
    Input:
        preds: B X T X C
    Output:
        scores: B
    """
    if ue == "vanilla":
        token_score, indices = torch.max(preds, dim=-1)
        ue = 1 - token_score # B X T
    elif ue == "entropy":
        ue = torch.sum(-preds * torch.log(torch.clip(preds, 1e-8, 1)), axis=-1)  # B X T
    else:
        raise ValueError("Unknown uncertainty estimation method.")

    return torch.mean(ue, dim=-1)  # B