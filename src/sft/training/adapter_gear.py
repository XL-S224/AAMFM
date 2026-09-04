import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, dispatch_model
from Bio.PDB import PDBParser
from esm.models.esm3 import ESM3
from esm.pretrained import (
    ESM3_function_decoder_v0,
    ESM3_sm_open_v0,
    ESM3_structure_decoder_v0,
    ESM3_structure_encoder_v0,
)
from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    GenerationConfig,
    SamplingConfig,
)
from esm.tokenization import EsmSequenceTokenizer, get_esm3_model_tokenizers
from esm.utils.constants.models import (
    ESM3_FUNCTION_DECODER_V0,
    ESM3_OPEN_SMALL,
    ESM3_STRUCTURE_DECODER_V0,
    ESM3_STRUCTURE_ENCODER_V0,
)
from esm.utils.structure.affine3d import (
    build_affine3d_from_coordinates,
)
from huggingface_hub import login
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class antigen_feature_adapter(nn.Module):
    def __init__(self, input_dim, interface_dim, bottleneck_dim):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        self.antigen_encoder = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            Permute(0, 2, 1),
            nn.Conv1d(1024, 512, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1),
        )
        self.antigen_proj = nn.Linear(512, bottleneck_dim)
        self.default_antigen = nn.Parameter(torch.randn(1, 512))

        self.embeddingLayer = nn.Embedding(2, 8)
        self.positionalEncodings = nn.Parameter(torch.randn(4000, 8))

        # MHA + FFN
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=8, nhead=8, dim_feedforward=128, dropout=0.4, batch_first=True
        )
        self.Prompt_encoder = nn.TransformerEncoder(encoder_layers, num_layers=1)

        self.mlp = nn.Sequential(
            nn.Linear(bottleneck_dim + 8, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
        )

        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.up_project = nn.Linear(bottleneck_dim, input_dim)
        self.activation = nn.GELU()

        self.gate_net = nn.Sequential(
            nn.Linear(bottleneck_dim + input_dim, bottleneck_dim), nn.Sigmoid()
        )
        nn.init.xavier_uniform_(self.embeddingLayer.weight)
        nn.init.normal_(self.positionalEncodings, mean=0, std=0.02)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, interface=None, antigen_features=None):
        """
        Docstring for forward
        
        :param self: class
        :param x: the original information
        :param interface: the mask of the antigen-features to present the interaction of each residue 
        :param antigen_features: precomputed antigen features
        """

        B, L, D = x.shape

        if antigen_features is not None:
            if antigen_features.dim() == 2:
                antigen_features = antigen_features.unsqueeze(0)
        
        # interface-cutting
        if interface.size(1) > L:
            interface = interface[:, :L]

        # using different  waves to present the interaction sites
        interface_embd = self.embeddingLayer(interface)  # [B, L, 8]

        seq_len = interface_embd.size(1)
        # max is 4000
        pos_enc = self.positionalEncodings[:seq_len, :]

        # merge the waves
        interface_embd = interface_embd + pos_enc.unsqueeze(0)  # [B, L, 8]

        interface_embd = self.Prompt_encoder(interface_embd)  # [B, L, 8]

        has_feat = antigen_features is not None
        feat_input = torch.zeros(B, 1, 3072, device=x.device) if not has_feat else antigen_features
        
        # FFN+GULE+LN+Pernumte+Conv1d+AvgPool1d        
        antigen_enc_out = self.antigen_encoder(feat_input).squeeze(-1)

        default_feat = self.default_antigen.expand(B, -1)
        m = float(has_feat)
        antigen_combined = m * antigen_enc_out + (1.0 - m) * default_feat
        antigen_feat = self.antigen_proj(antigen_combined)

        default_antigen = self.default_antigen.expand(B, -1)  # for DDP training
        dummy_input = torch.zeros((B, 1, 3072), device=x.device)

        # FFN+GULE+LN+Pernumte+Conv1d+AvgPool1d        
        antigen = self.antigen_encoder(antigen_features if antigen_features is not None else dummy_input)
        antigen = antigen.squeeze(-1)
        radio = float(antigen_features is None)
        antigen = antigen + radio * default_antigen.sum()
        antigen_feat = self.antigen_proj(antigen)

        x_down = self.down_project(x)

        # cat the interface embedding in D
        x_cat = torch.cat([x_down, interface_embd], dim=-1)

        x_processed = self.dropout(self.mlp(x_cat))
        gate_input = torch.cat([x_processed, x], dim=-1)
        gate = self.gate_net(gate_input)

        modulated = gate * antigen_feat.unsqueeze(1)
        output = x + self.dropout(self.up_project(modulated))
        return output


class ModifiedLastBlock(nn.Module):
    def __init__(self, interface_dim=8, adapter_dim=64):
        super().__init__()
        self.adapter = antigen_feature_adapter(
            1536, interface_dim, adapter_dim
        )  # Corrected input_dim

    def forward(self, x, interface=None, antigen_feat=None):
        if interface is not None:
            x = self.adapter(x, interface, antigen_feat)
        return x


from esm.layers.transformer_stack import TransformerStack
from esm.models.esm3 import ESMOutput
from esm.utils.constants import esm3 as C
from esm.utils.structure.affine3d import Affine3D


class Modified_transformer(TransformerStack):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter = ModifiedLastBlock(interface_dim=8, adapter_dim=64)

    def forward(
        self,
        x: torch.Tensor,
        sequence_id: torch.Tensor | None = None,
        affine: Affine3D | None = None,
        affine_mask: torch.Tensor | None = None,
        chain_id: torch.Tensor | None = None,
        interface: torch.Tensor | None = None,
        antigen_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the TransformerStack.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, d_model).
            sequence_id (torch.Tensor): The sequence ID tensor of shape (batch_size, sequence_length).
            affine (Affine3D | None): The affine transformation tensor or None.
            affine_mask (torch.Tensor | None): The affine mask tensor or None.
            chain_id (torch.Tensor): The protein chain tensor of shape (batch_size, sequence_length).
                Only used in geometric attention.

        Returns:
            post_norm: The output tensor of shape (batch_size, sequence_length, d_model).
            pre_norm: The embedding of shape (batch_size, sequence_length, d_model).
        """
        *batch_dims, _ = x.shape
        if chain_id is None:
            chain_id = torch.ones(size=batch_dims, dtype=torch.int64, device=x.device)
        for block in self.blocks:
            if x.dim() == 3:
                x = block(x, sequence_id, affine, affine_mask, chain_id)
            elif x.dim() == 4:
                x = block(x.squeeze(0), sequence_id, affine, affine_mask, chain_id)
        x = self.adapter(x, interface, antigen_feat)
        return self.norm(x), x


import contextlib
from functools import partial
from typing import Callable

import attr
import einops
import torch
import torch.nn as nn
from attr import dataclass
from esm.layers.regression_head import RegressionHead
from esm.layers.transformer_stack import TransformerStack
from esm.models.function_decoder import FunctionTokenDecoder
from esm.models.vqvae import (
    StructureTokenDecoder,
    StructureTokenEncoder,
)
from esm.sdk.api import (
    ESM3InferenceClient,
    ESMProtein,
    ESMProteinTensor,
    ForwardAndSampleOutput,
    ForwardTrackData,
    GenerationConfig,
    LogitsConfig,
    LogitsOutput,
    ProteinType,
    SamplingConfig,
)
from esm.tokenization import TokenizerCollectionProtocol
from esm.utils import encoding
from esm.utils.constants import esm3 as C
from esm.utils.constants.models import (
    ESM3_OPEN_SMALL,
    normalize_model_name,
)
from esm.utils.decoding import decode_protein_tensor
from esm.utils.misc import rbf
from esm.utils.sampling import (
    _BatchedESMProteinTensor,
    get_default_sampling_config,
    validate_sampling_config,
)
from esm.utils.structure.affine3d import (
    build_affine3d_from_coordinates,
)
from .generation_sub import (
    _batch_forward,
    _sample_per_prompt,
    _slice_tensor_dataclass,
    iterative_sampling_raw,
    iterative_sampling_tokens,
)


class ESM3Wrapper(ESM3):
    def __init__(self, base_model):
        super().__init__(
            d_model=1536,
            n_heads=24,
            v_heads=256,
            n_layers=48,
            structure_encoder_fn=ESM3_structure_encoder_v0,
            structure_decoder_fn=ESM3_structure_decoder_v0,
            function_decoder_fn=ESM3_function_decoder_v0,
            tokenizers=get_esm3_model_tokenizers(ESM3_OPEN_SMALL),
        )
        # Load the pretrained transformer from base_model
        self.transformer = base_model.transformer

        # Replace with Modified_transformer that preserves the pretrained weights
        modified_transformer = Modified_transformer(
            1536,
            24,
            256,
            48,
            mask_and_zero_frameless=True,
        )

        # Copy pretrained weights from original transformer to modified one
        modified_transformer.load_state_dict(
            self.transformer.state_dict(), strict=False
        )
        self.transformer = modified_transformer

    def forward(
        self,
        *,
        sequence_tokens: torch.Tensor | None = None,
        structure_tokens: torch.Tensor | None = None,
        ss8_tokens: torch.Tensor | None = None,
        sasa_tokens: torch.Tensor | None = None,
        function_tokens: torch.Tensor | None = None,
        residue_annotation_tokens: torch.Tensor | None = None,
        average_plddt: torch.Tensor | None = None,
        per_res_plddt: torch.Tensor | None = None,
        structure_coords: torch.Tensor | None = None,
        chain_id: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        interface: torch.Tensor | None = None,
        antigen_feat: torch.Tensor | None = None,
        # **kwargs
    ) -> ESMOutput:
        """
        Performs forward pass through the ESM3 model. Check utils to see how to tokenize inputs from raw data.

        Args:
            sequence_tokens (torch.Tensor, optional): The amino acid tokens.
            structure_tokens (torch.Tensor, optional): The structure tokens.
            ss8_tokens (torch.Tensor, optional): The secondary structure tokens.
            sasa_tokens (torch.Tensor, optional): The solvent accessible surface area tokens.
            function_tokens (torch.Tensor, optional): The function tokens.
            residue_annotation_tokens (torch.Tensor, optional): The residue annotation tokens.
            average_plddt (torch.Tensor, optional): The average plddt across the entire sequence.
            per_res_plddt (torch.Tensor, optional): The per residue plddt, if you want to specify exact plddts, use this,
                otherwise, use average_plddt.
            structure_coords (torch.Tensor, optional): The structure coordinates, in the form of (B, L, 3, 3).
            chain_id (torch.Tensor, optional): The chain ID
            sequence_id (torch.Tensor, optional): The sequence ID.

        Returns:
            ESMOutput: The output of the ESM3 model.

        Raises:
            ValueError: If at least one of the inputs is None.

        """
        # Reasonable defaults:
        try:
            L, device = next(
                (x.shape[1], x.device)
                for x in [
                    sequence_tokens,
                    structure_tokens,
                    ss8_tokens,
                    sasa_tokens,
                    structure_coords,
                    function_tokens,
                    residue_annotation_tokens,
                ]
                if x is not None
            )
        except StopIteration:
            raise ValueError("At least one of the inputs must be non-None")

        t = self.tokenizers
        defaults = lambda x, tok: (
            torch.full((1, L), tok, dtype=torch.long, device=device) if x is None else x
        )
        sequence_tokens = defaults(sequence_tokens, t.sequence.mask_token_id)
        ss8_tokens = defaults(ss8_tokens, C.SS8_PAD_TOKEN)
        sasa_tokens = defaults(sasa_tokens, C.SASA_PAD_TOKEN)
        average_plddt = defaults(average_plddt, 1).float()
        per_res_plddt = defaults(per_res_plddt, 0).float()
        chain_id = defaults(chain_id, 0)
        interface = defaults(interface, 0)
        if residue_annotation_tokens is None:
            residue_annotation_tokens = torch.full(
                (1, L, 16), C.RESIDUE_PAD_TOKEN, dtype=torch.long, device=device
            )

        if function_tokens is None:
            function_tokens = torch.full(
                (1, L, 8), C.INTERPRO_PAD_TOKEN, dtype=torch.long, device=device
            )

        if structure_coords is None:
            structure_coords = torch.full(
                (1, L, 3, 3), float("nan"), dtype=torch.float, device=device
            )

        structure_coords = structure_coords[
            ..., :3, :
        ]  # In case we pass in an atom14 or atom37 repr
        affine, affine_mask = build_affine3d_from_coordinates(structure_coords)

        structure_tokens = defaults(structure_tokens, C.STRUCTURE_MASK_TOKEN)
        assert structure_tokens is not None
        structure_tokens = (
            structure_tokens.masked_fill(structure_tokens == -1, C.STRUCTURE_MASK_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_BOS_TOKEN, C.STRUCTURE_BOS_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_PAD_TOKEN, C.STRUCTURE_PAD_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_EOS_TOKEN, C.STRUCTURE_EOS_TOKEN)
            .masked_fill(
                sequence_tokens == C.SEQUENCE_CHAINBREAK_TOKEN,
                C.STRUCTURE_CHAINBREAK_TOKEN,
            )
        )

        x = self.encoder(
            sequence_tokens,
            structure_tokens,
            average_plddt,
            per_res_plddt,
            ss8_tokens,
            sasa_tokens,
            function_tokens,
            residue_annotation_tokens,
        )
        x, embedding = self.transformer(
            x, sequence_id, affine, affine_mask, chain_id, interface, antigen_feat
        )
        return self.output_heads(x, embedding)

    def logits(
        self,
        input: ESMProteinTensor | _BatchedESMProteinTensor,
        interface: torch.Tensor | None = None,
        antigen_feat: torch.Tensor | None = None,
        config: LogitsConfig = LogitsConfig(),
    ) -> LogitsOutput:
        if not isinstance(input, _BatchedESMProteinTensor):
            # Create batch dimension if necessary.
            input = _BatchedESMProteinTensor.from_protein_tensor(input)

        device = torch.device(input.device)

        # Default plddt conditioning for inference. 1s where coordinates are provided.
        if input.coordinates is None:
            per_res_plddt = None
        else:
            # 1.0 if all coordinates at specific indices have valid non-nan values.
            per_res_plddt = input.coordinates.isfinite().all(dim=-1).any(dim=-1).float()

        with (
            torch.no_grad(),  # Assume no gradients for now...
            torch.autocast(enabled=True, device_type=device.type, dtype=torch.bfloat16)  # type: ignore
            if device.type == "cuda"
            else contextlib.nullcontext(),
        ):
            output = self.forward(
                sequence_tokens=input.sequence,
                structure_tokens=input.structure,
                ss8_tokens=input.secondary_structure,
                sasa_tokens=input.sasa,
                function_tokens=input.function,
                residue_annotation_tokens=input.residue_annotations,
                average_plddt=torch.tensor(1.0, device=input.device),
                per_res_plddt=per_res_plddt,
                structure_coords=input.coordinates,
                chain_id=None,
                sequence_id=None,
                interface=interface,
                antigen_feat=antigen_feat,
            )

        output = ESMOutput(
            **{k: v.to(device).to(torch.float32) for k, v in vars(output).items()}
        )

        return LogitsOutput(
            logits=ForwardTrackData(
                sequence=output.sequence_logits if config.sequence else None,
                structure=output.structure_logits if config.structure else None,
                secondary_structure=output.secondary_structure_logits
                if config.secondary_structure
                else None,
                sasa=output.sasa_logits if config.sasa else None,
                function=output.function_logits if config.function else None,
            ),
            residue_annotation_logits=output.residue_logits
            if config.residue_annotations
            else None,
            embeddings=output.embeddings if config.return_embeddings else None,
        )

    # The following methods are for the ESM3InferenceClient interface
    def generate(
        self,
        input: ProteinType,
        config: GenerationConfig,
        interface: torch.Tensor,
        antigen_feat: torch.Tensor,
    ) -> ProteinType:
        """Wrap around batched generation."""
        proteins = self.batch_generate([input], [config], [interface], [antigen_feat])
        assert len(proteins) == 1
        return proteins[0]

    def batch_generate(
        self,
        inputs: list[ProteinType],
        configs: list[GenerationConfig],
        interface: list[torch.Tensor],
        antigen_feat: list[torch.Tensor],
    ) -> list[ProteinType]:
        assert len(inputs) == len(configs), (
            "Must have the same number of prompts and configs."
        )

        if inputs == []:
            # Nothing to do.
            return []

        # Make sure prompts are of the same type.
        t = type(inputs[0])
        for i in range(1, len(inputs)):
            assert isinstance(inputs[i], t), (
                "Prompts must have the same type. Got "
                f"{t.__name__ and type(inputs[i]).__name__} instead."
            )

        if isinstance(inputs[0], ESMProtein):
            return iterative_sampling_raw(
                self, inputs, configs, interface, antigen_feat
            )  # type: ignore
        elif isinstance(inputs[0], ESMProteinTensor):
            return iterative_sampling_tokens(
                self,
                inputs,  # type: ignore
                configs,
                self.tokenizers,  # type: ignore
                interface,
                antigen_feat,
            )
        else:
            raise ValueError("Input must be an ESMProtein or ESMProteinTensor")

    def encode(self, input: ESMProtein) -> ESMProteinTensor:
        input = attr.evolve(input)  # Make a copy

        sequence_tokens = None
        structure_tokens = None
        secondary_structure_tokens = None
        sasa_tokens = None
        function_tokens = None
        residue_annotation_tokens = None

        coordinates = None

        if input.sequence is not None:
            sequence_tokens = encoding.tokenize_sequence(
                input.sequence, self.tokenizers.sequence, add_special_tokens=True
            )
        if input.secondary_structure is not None:
            secondary_structure_tokens = encoding.tokenize_secondary_structure(
                input.secondary_structure,
                self.tokenizers.secondary_structure,
                add_special_tokens=True,
            )
        if input.sasa is not None:
            sasa_tokens = encoding.tokenize_sasa(
                input.sasa, self.tokenizers.sasa, add_special_tokens=True
            )

        # Infer input length
        sequence_length = -1
        if sequence_tokens is not None:
            sequence_length = len(sequence_tokens)
        elif secondary_structure_tokens is not None:
            sequence_length = len(secondary_structure_tokens)
        elif sasa_tokens is not None:
            sequence_length = len(sasa_tokens)

        # Try to infer input length from structure data
        if input.coordinates is not None:
            coordinates, _, structure_tokens = encoding.tokenize_structure(
                input.coordinates,
                self.get_structure_encoder(),
                structure_tokenizer=self.tokenizers.structure,
                reference_sequence=input.sequence or "",
                add_special_tokens=True,
            )
            if sequence_length == -1:
                sequence_length = len(structure_tokens)

        if sequence_length == -1:
            raise ValueError(
                "Cannot infer input length from input data. Please provide one of: sequence, structure, secondary_structure, sasa.\n"
                "To condition on sequence length only, use ESM3LocalInferenceClient.get_default_sequence(sequence_length) to generate a default sequence input."
            )

        # Function and Residue annotations
        if input.function_annotations is not None:
            if input.sequence is None:
                reference_sequence = encoding.get_default_sequence(sequence_length - 2)
            else:
                reference_sequence = input.sequence
            (function_tokens, residue_annotation_tokens) = (
                encoding.tokenize_function_annotations(
                    input.function_annotations,
                    reference_sequence=reference_sequence,
                    function_tokenizer=self.tokenizers.function,
                    residue_annotation_tokenizer=self.tokenizers.residue_annotations,
                    add_special_tokens=True,
                )
            )

        return ESMProteinTensor(
            sequence=sequence_tokens,
            structure=structure_tokens,
            secondary_structure=secondary_structure_tokens,
            sasa=sasa_tokens,
            function=function_tokens,
            residue_annotations=residue_annotation_tokens,
            coordinates=coordinates,
        ).to(next(self.parameters()).device)

    def decode(self, input: ESMProteinTensor) -> ESMProtein:
        return decode_protein_tensor(
            input=input,
            tokenizers=self.tokenizers,
            structure_token_decoder=self.get_structure_decoder(),
            function_token_decoder=self.get_function_decoder(),
        )

    def forward_and_sample(
        self,
        input: ESMProteinTensor,
        sampling_configuration: SamplingConfig,
        interface: torch.Tensor,
        antigen_feat: torch.Tensor,
    ) -> ForwardAndSampleOutput:
        validate_sampling_config(sampling_configuration, on_invalid="warn")

        protein_tensor = attr.evolve(input)  # Make a copy

        device = next(self.parameters()).device

        sampling_config = sampling_configuration
        if sampling_config is None:
            sampling_config = get_default_sampling_config(self.tokenizers)

        # Initialize default values for missing tracks
        default_protein_tensor = ESMProteinTensor.empty(
            len(input) - 2, tokenizers=self.tokenizers, device=input.device
        )
        for track in attr.fields(ESMProteinTensor):
            if getattr(protein_tensor, track.name, None) is None:
                setattr(
                    protein_tensor,
                    track.name,
                    getattr(default_protein_tensor, track.name, None),
                )

        if len(protein_tensor) <= 0:
            raise ValueError("No input data provided")

        # Move input protein to proper device.
        batched_protein = _BatchedESMProteinTensor.from_protein_tensor(protein_tensor)
        batched_protein.to(device)

        logits_output: LogitsOutput = _batch_forward(self, batched_protein)
        forward_and_sample_out: ForwardAndSampleOutput = _sample_per_prompt(
            batched_protein, logits_output, sampling_config, self.tokenizers
        )

        # There is only 1 prompt to sample for.
        return _slice_tensor_dataclass(forward_and_sample_out, 0)

        # There is only 1 prompt to sample for.
        return _slice_tensor_dataclass(forward_and_sample_out, 0)


def freeze_model_except_adapter(model):
    for name, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.transformer.adapter.named_parameters():
        param.requires_grad = True
