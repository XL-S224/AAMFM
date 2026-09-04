from __future__ import annotations

import math
import random

import torch
from torch.distributions.beta import Beta


def _cosine_schedule(t: torch.Tensor) -> torch.Tensor:
    return torch.cos(t * math.pi * 0.5)


def _sample_random_mask_prob(
    *,
    device: torch.device,
    random_mask_prob: float | None,
    random_beta: tuple[float, float],
) -> float:
    if random_mask_prob is not None:
        return random_mask_prob
    if random.random() < 0.8:
        alpha, beta = random_beta
        dist = Beta(
            torch.tensor([alpha], device=device), torch.tensor([beta], device=device)
        )
        return dist.sample().item()
    return random.random()


def _chain_mask(chain_id: torch.Tensor, h_id: torch.Tensor, l_id: torch.Tensor) -> torch.Tensor:
    h_id = h_id.to(chain_id.device)
    l_id = l_id.to(chain_id.device)
    h_id_expanded = h_id.unsqueeze(1).expand_as(chain_id)
    l_id_expanded = l_id.unsqueeze(1).expand_as(chain_id)
    return (chain_id == h_id_expanded) | (chain_id == l_id_expanded)


def mask_tokens(tokens, mask_idx, pad_idx, pipe_idx=1):
    masked_tokens = tokens.clone()
    mask = (tokens != pad_idx) & (tokens != pipe_idx)
    mask_prob = _sample_random_mask_prob(
        device=tokens.device,
        random_mask_prob=None,
        random_beta=(3.0, 9.0),
    )
    random_mask = torch.rand(tokens.shape, device=tokens.device) < mask_prob
    masked_tokens[mask & random_mask] = mask_idx
    return masked_tokens


def mask_stru_tokens(tokens, mask_idx, special_tokens):
    masked_tokens = tokens.clone()
    mask = ~torch.isin(tokens, torch.tensor(special_tokens, device=tokens.device))
    t = torch.rand(1, device=tokens.device)
    mask_prob = _cosine_schedule(t).item()
    random_mask = torch.rand(tokens.shape, device=tokens.device) < mask_prob
    masked_tokens[mask & random_mask] = mask_idx
    return masked_tokens


def mask_antibody_seq(tokens, mask_idx, pad_idx, chain_id, h_id, l_id, pipe_idx=31):
    masked_tokens = tokens.clone()
    mask = (tokens != pad_idx) & (tokens != pipe_idx)
    chain_mask = _chain_mask(chain_id, h_id, l_id)
    masked_tokens[mask & chain_mask] = mask_idx
    return masked_tokens


def mask_antibody_seq_cdr(tokens, cdr_pos, mask_idx, pad_idx, chain_id, h_id, l_id, pipe_idx=31):
    masked_tokens = tokens.clone()
    mask = (tokens != pad_idx) & (tokens != pipe_idx)
    chain_mask = _chain_mask(chain_id, h_id, l_id)
    cdr_mask = cdr_pos != 0
    masked_tokens[mask & chain_mask & cdr_mask] = mask_idx
    return masked_tokens


def mask_seq_single_cdr(
    tokens, cdr_pos, cdr_index, mask_idx, pad_idx, chain_id, h_id, l_id, pipe_idx=31
):
    masked_tokens = tokens.clone()
    mask = (tokens != pad_idx) & (tokens != pipe_idx)
    chain_mask = _chain_mask(chain_id, h_id, l_id)
    cdr_mask = cdr_pos == cdr_index
    masked_tokens[mask & chain_mask & cdr_mask] = mask_idx
    return masked_tokens


def mask_antibody_stru_cdr(tokens, cdr_pos, mask_idx, special_tokens, chain_id, h_id, l_id):
    masked_tokens = tokens.clone()
    mask = ~torch.isin(tokens, torch.tensor(special_tokens, device=tokens.device))
    chain_mask = _chain_mask(chain_id, h_id, l_id)
    cdr_mask = cdr_pos != 0
    masked_tokens[mask & chain_mask & cdr_mask] = mask_idx
    return masked_tokens


def mask_stru_single_cdr(tokens, cdr_pos, cdr_index, mask_idx, special_tokens, chain_id, h_id, l_id):
    masked_tokens = tokens.clone()
    mask = ~torch.isin(tokens, torch.tensor(special_tokens, device=tokens.device))
    chain_mask = _chain_mask(chain_id, h_id, l_id)
    cdr_mask = cdr_pos == cdr_index
    masked_tokens[mask & chain_mask & cdr_mask] = mask_idx
    return masked_tokens


def random_mask_antibody_seq_tokens(
    tokens,
    mask_idx,
    pad_idx,
    chain_id,
    h_id,
    l_id,
    pipe_idx=31,
    random_mask_prob: float | None = None,
    random_beta: tuple[float, float] = (3.0, 9.0),
):
    masked_tokens = tokens.clone()
    mask = (tokens != pad_idx) & (tokens != pipe_idx)
    chain_mask = _chain_mask(chain_id, h_id, l_id)
    mask_prob = _sample_random_mask_prob(
        device=tokens.device,
        random_mask_prob=random_mask_prob,
        random_beta=random_beta,
    )
    random_mask = torch.rand(tokens.shape, device=tokens.device) < mask_prob
    masked_tokens[mask & chain_mask & random_mask] = mask_idx
    return masked_tokens


def random_mask_antibody_seq_cdr_tokens(
    tokens,
    cdr_pos,
    mask_idx,
    pad_idx,
    chain_id,
    h_id,
    l_id,
    pipe_idx=31,
    random_mask_prob: float | None = None,
    random_beta: tuple[float, float] = (3.0, 9.0),
):
    masked_tokens = tokens.clone()
    mask = (tokens != pad_idx) & (tokens != pipe_idx)
    chain_mask = _chain_mask(chain_id, h_id, l_id)
    cdr_mask = cdr_pos != 0
    mask_prob = _sample_random_mask_prob(
        device=tokens.device,
        random_mask_prob=random_mask_prob,
        random_beta=random_beta,
    )
    random_mask = torch.rand(tokens.shape, device=tokens.device) < mask_prob
    masked_tokens[mask & chain_mask & cdr_mask & random_mask] = mask_idx
    return masked_tokens


def random_mask_antibody_seq_single_cdr_tokens(
    tokens,
    cdr_pos,
    cdr_index,
    mask_idx,
    pad_idx,
    chain_id,
    h_id,
    l_id,
    pipe_idx=31,
    random_mask_prob: float | None = None,
    random_beta: tuple[float, float] = (3.0, 9.0),
):
    masked_tokens = tokens.clone()
    mask = (tokens != pad_idx) & (tokens != pipe_idx)
    chain_mask = _chain_mask(chain_id, h_id, l_id)
    cdr_mask = cdr_pos == cdr_index
    mask_prob = _sample_random_mask_prob(
        device=tokens.device,
        random_mask_prob=random_mask_prob,
        random_beta=random_beta,
    )
    random_mask = torch.rand(tokens.shape, device=tokens.device) < mask_prob
    masked_tokens[mask & chain_mask & cdr_mask & random_mask] = mask_idx
    return masked_tokens


def random_mask_antibody_stru_token(
    tokens,
    mask_idx,
    special_tokens,
    chain_id,
    h_id,
    l_id,
):
    masked_tokens = tokens.clone()
    mask = ~torch.isin(tokens, torch.tensor(special_tokens, device=tokens.device))
    chain_mask = _chain_mask(chain_id, h_id, l_id)
    t = torch.rand(1, device=tokens.device)
    mask_prob = _cosine_schedule(t).item()
    random_mask = torch.rand(tokens.shape, device=tokens.device) < mask_prob
    masked_tokens[mask & chain_mask & random_mask] = mask_idx
    return masked_tokens


def random_mask_antibody_stru_cdr_token(
    tokens,
    cdr_pos,
    mask_idx,
    special_tokens,
    chain_id,
    h_id,
    l_id,
):
    masked_tokens = tokens.clone()
    mask = ~torch.isin(tokens, torch.tensor(special_tokens, device=tokens.device))
    chain_mask = _chain_mask(chain_id, h_id, l_id)
    cdr_mask = cdr_pos != 0
    t = torch.rand(1, device=tokens.device)
    mask_prob = _cosine_schedule(t).item()
    random_mask = torch.rand(tokens.shape, device=tokens.device) < mask_prob
    masked_tokens[mask & chain_mask & cdr_mask & random_mask] = mask_idx
    return masked_tokens


def random_mask_antibody_stru_single_cdr_token(
    tokens,
    cdr_pos,
    cdr_index,
    mask_idx,
    special_tokens,
    chain_id,
    h_id,
    l_id,
):
    masked_tokens = tokens.clone()
    mask = ~torch.isin(tokens, torch.tensor(special_tokens, device=tokens.device))
    chain_mask = _chain_mask(chain_id, h_id, l_id)
    cdr_mask = cdr_pos == cdr_index
    t = torch.rand(1, device=tokens.device)
    mask_prob = _cosine_schedule(t).item()
    random_mask = torch.rand(tokens.shape, device=tokens.device) < mask_prob
    masked_tokens[mask & chain_mask & cdr_mask & random_mask] = mask_idx
    return masked_tokens
