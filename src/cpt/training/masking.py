from __future__ import annotations

import math
import random

import torch
from torch.distributions.beta import Beta


def mask_tokens(tokens, mask_idx, pad_idx, pipe_idx=1, return_mask=False):
    masked_tokens = tokens.clone()
    mask = (tokens != pad_idx) & (tokens != pipe_idx)

    if random.random() < 0.8:
        beta_dist = Beta(torch.tensor([3.0], device=tokens.device), torch.tensor([9.0], device=tokens.device))
        mask_prob = beta_dist.sample().item()
    else:
        mask_prob = random.random()

    random_mask = torch.rand(tokens.shape, device=tokens.device) < mask_prob
    final_mask = mask & random_mask
    masked_tokens[final_mask] = mask_idx
    if return_mask:
        return masked_tokens, final_mask
    return masked_tokens


def cosine_schedule(t: torch.Tensor):
    return torch.cos(t * math.pi * 0.5)


def mask_stru_tokens(tokens, mask_idx, special_tokens, return_mask=False):
    masked_tokens = tokens.clone()
    mask = ~torch.isin(tokens, torch.tensor(special_tokens, device=tokens.device))

    t = torch.rand(1, device=tokens.device)
    mask_prob = cosine_schedule(t).item()

    random_mask = torch.rand(tokens.shape, device=tokens.device) < mask_prob
    final_mask = mask & random_mask
    masked_tokens[final_mask] = mask_idx
    if return_mask:
        return masked_tokens, final_mask
    return masked_tokens


def mask_antibody_seq_cdr(tokens, cdr_pos, mask_idx, pad_idx, chain_id, h_id, l_id, pipe_idx=31, return_mask=False):
    cdr_mask = cdr_pos != 0
    masked_tokens = tokens.clone()
    mask = (tokens != pad_idx) & (tokens != pipe_idx)

    h_id_expanded = h_id.unsqueeze(1).expand_as(chain_id)
    l_id_expanded = l_id.unsqueeze(1).expand_as(chain_id)
    chain_mask = (chain_id == h_id_expanded) | (chain_id == l_id_expanded)

    final_mask = mask & chain_mask & cdr_mask
    masked_tokens[final_mask] = mask_idx
    if return_mask:
        return masked_tokens, final_mask
    return masked_tokens


def mask_seq_framework(tokens, cdr_pos, mask_idx, pad_idx, chain_id, h_id, l_id, pipe_idx=31, return_mask=False):
    masked_tokens = tokens.clone()
    mask = (tokens != pad_idx) & (tokens != pipe_idx)

    h_id_expanded = h_id.unsqueeze(1).expand_as(chain_id)
    l_id_expanded = l_id.unsqueeze(1).expand_as(chain_id)
    chain_mask = (chain_id == h_id_expanded) | (chain_id == l_id_expanded)
    framework_mask = cdr_pos == 0

    final_mask = mask & chain_mask & framework_mask
    masked_tokens[final_mask] = mask_idx
    if return_mask:
        return masked_tokens, final_mask
    return masked_tokens


def mask_antibody_stru_cdr(tokens, cdr_pos, mask_idx, special_tokens, chain_id, h_id, l_id, return_mask=False):
    masked_tokens = tokens.clone()
    mask = ~torch.isin(tokens, torch.tensor(special_tokens, device=tokens.device))

    h_id_expanded = h_id.unsqueeze(1).expand_as(chain_id)
    l_id_expanded = l_id.unsqueeze(1).expand_as(chain_id)
    chain_mask = (chain_id == h_id_expanded) | (chain_id == l_id_expanded)
    cdr_mask = cdr_pos != 0

    final_mask = mask & chain_mask & cdr_mask
    masked_tokens[final_mask] = mask_idx
    if return_mask:
        return masked_tokens, final_mask
    return masked_tokens


def mask_stru_framework(tokens, cdr_pos, mask_idx, special_tokens, chain_id, h_id, l_id, return_mask=False):
    masked_tokens = tokens.clone()
    mask = ~torch.isin(tokens, torch.tensor(special_tokens, device=tokens.device))

    h_id_expanded = h_id.unsqueeze(1).expand_as(chain_id)
    l_id_expanded = l_id.unsqueeze(1).expand_as(chain_id)
    chain_mask = (chain_id == h_id_expanded) | (chain_id == l_id_expanded)
    framework_mask = cdr_pos == 0

    final_mask = mask & chain_mask & framework_mask
    masked_tokens[final_mask] = mask_idx
    if return_mask:
        return masked_tokens, final_mask
    return masked_tokens
