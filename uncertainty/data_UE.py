import torch


def data_uncertainty(preds, ue="vanilla"):
    """
    Input:
        preds: B X T X C
    Output:
        scores: B
    """
    if ue == "vanilla":
        token_score = 1 - torch.max(preds, dim=-1)[0]  # B X T
    elif ue == "entropy":
        token_score = torch.sum(-preds * torch.log(torch.clip(preds, 1e-8, 1)), axis=-1)  # B X T
    else:
        raise ValueError("Unknown uncertainty estimation method.")

    return torch.mean(token_score, dim=-1)  # B