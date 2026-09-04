import os
import random
import logging
import datetime
import pandas as pd
import joblib
import pickle
import subprocess
import torch
import Bio
from Bio import PDB, SeqRecord, SeqIO, Seq
from Bio.PDB import PDBExceptions
from Bio.PDB import Polypeptide
from Bio import PDB, SeqRecord, SeqIO, Seq
# Import necessary classes explicitly
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from typing import Dict, List
import numpy as np
import torch.nn.functional as F
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.structure.protein_complex import ProteinComplex
from esm.utils.structure.normalize_coordinates import (
    apply_frame_to_coords,
    get_protein_normalization_frame,
    normalize_coordinates,
)
from esm.pretrained import (
    ESM3_function_decoder_v0,
    ESM3_sm_open_v0,
    ESM3_structure_decoder_v0,
    ESM3_structure_encoder_v0,
)
from esm.tokenization import EsmSequenceTokenizer, StructureTokenizer
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from torch import nn, optim


ALLOWED_AG_TYPES = {
    'protein',
    'protein | protein',
    'protein | protein | protein',
    'protein | protein | protein | protein | protein',
    'protein | protein | protein | protein',
}

RESOLUTION_THRESHOLD = 5.0

RAbD_IDS = [
    "1a14", "1a2y", "1fe8", "1ic7", "1iqd", "1n8z", "1ncb", "1osp", "1uj3", "1w72",
    "2adf", "2b2x", "2cmr", "2dd8", "2ghw", "2vxt", "2xqy", "2xwt", "2ypv", "3bn9",
    "3cx5", "3ffd", "3hi6", "3k2u", "3l95", "3mxw", "3nid", "3o2d", "3rkd",
    "3s35", "3w9e", "4cmh", "4dtg", "4dvr", "4ffv", "4fqj", "4g6j",
    "4g6m", "4h8w", "4ki5", "4lvn", "4ot1", "4qci", "4xnq", "4ydk", "5b8c", "5bv7",
    "5d93", "5en2", "5f9o", "5ggs", "5hi4", "5j13", "5l6y", "5mes", "5nuz"
]

IGGM_IDS =[
    "8fxc", "8d4r", "8k5g", "8iv5", "8g4p", "8h7z", "8szy", "8j7y", "8iv4", "8wsq",
    "8g4t", "8dpl", "8fdo", "8hpu", "8dyy", "8iv8", "8g4p", "8t04", "8dz3", "8t9z",
    "7trh", "8ucd", "8ix3", "8bg5", "8r8d", "8gh4", "8szy", "8eay", "8hrd", "7yv1",
    "8gsp", "8b7h", "8dpz", "8dyw", "7yh6", "8j7e", "8pq2", "8ezl", "8ded", "8eee",
    "8ez7", "8tco", "8d0y", "8udg", "8ezm", "8e6j", "8awm", "8dun", "8smi", "8byu",
    "8e6k", "8f5i", "8tea", "8hrd", "8tea", "8gye", "8ee5", "8grr", "8x0t", "8fsj"
]


TEST_ANTIGENS = [
    # 'sars-cov-2 receptor binding domain',
    # 'hiv-1 envelope glycoprotein gp160',
    # 'mers s',
    # 'influenza a virus',
    # 'cd27 antigen',
    'programmed cell death protein 1'
    ]

tokenizer = EsmSequenceTokenizer()
encoder = ESM3_structure_encoder_v0().cuda()

def nan_to_empty_string(val):
    """
    Convert NaN or empty values to an empty string.
    """
    if val != val or not val:
        return ''
    else:
        return val


def nan_to_none(val):
    """
    Convert NaN or empty values to None.
    """
    if val != val or not val:
        return None
    else:
        return val


def split_sabdab_delimited_str(val):
    """
    Strip the delimiter and return a list.
    """
    if not val:
        return []
    else:
        return [s.strip() for s in val.split('|')]


def parse_sabdab_resolution(val):
    """
    Parse the resolution, handling special formats, and return it as a float.
    """
    if val == 'NOT' or not val or val != val:
        return None
    elif isinstance(val, str) and ',' in val:
        return float(val.split(',')[0].strip())
    else:
        return float(val)


def to_structure_encoder_inputs(
        complex,
        should_normalize_coordinates: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Data format normalization:
    if should_normalize_coordinates = Ture:
        Represent heavy-atom positions as relative positions.
    """

    coords = torch.tensor(complex.atom37_positions, dtype=torch.float32)
    plddt = torch.tensor(complex.confidence, dtype=torch.float32)
    residue_index = torch.tensor(complex.residue_index, dtype=torch.long)

    if should_normalize_coordinates:
        coords = normalize_coordinates(coords)
    return coords.unsqueeze(0), plddt.unsqueeze(0), residue_index.unsqueeze(0)


#     """

#     Args:

#     Returns:
#     """


















def cut_antibody(task):
    """
    Extracts the antibody part from a PDB file (truncated to residue numbers H_max and L_max) and includes the antigen chains.
    If the target file already exists, add chains from the current task that are not yet included.
    All chains from the same original PDB are saved in the same _cut.pdb file.

    Args:
        task: dict containing pdb_path, entry and related information

    Returns:
        str: path to the saved truncated PDB file
    """
    if not task or 'pdb_path' not in task or 'entry' not in task:
        raise ValueError("Incomplete task parameters; pdb_path and entry are required")

    pdb_path = task['pdb_path']
    entry = task['entry']
    pdb_code = entry.get('pdbcode', 'unknown')

    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"Original PDB file not found: {pdb_path}")

    H_id = entry.get('H_chain')
    L_id = entry.get('L_chain')
    antigen_ids = entry.get('ag_chains', [])

    if not H_id and not L_id and not antigen_ids:
         logging.warning(f"PDB: {pdb_code}, Entry: {entry.get('id', 'N/A')} - no valid chain IDs provided (H={H_id}, L={L_id}, Ag={antigen_ids}), skipping this entry.")
         pdb_path_cut = pdb_path[:-4] + '_cut.pdb'
         return pdb_path_cut

    H_max = 128
    L_max = 128

    pdb_path_cut = pdb_path[:-4] + '_cut.pdb'

    parser = Bio.PDB.PDBParser(QUIET=True)
    io = Bio.PDB.PDBIO()

    original_structure = None

    try:
        if not os.path.exists(pdb_path_cut):
            # --- Case 1: Output file does NOT exist ---
            logging.info(f"PDB: {pdb_code}, Entry: {entry.get('id', 'N/A')} - target file {pdb_path_cut} does not exist, creating a new file.")

            original_structure = parser.get_structure('original', pdb_path)
            original_model = original_structure[0]

            new_structure = Structure('cut_pdb')
            new_model = Model(0)
            new_structure.add(new_model)

            chains_added_this_run = set()

            if H_id and H_id in original_model:
                H_chain = original_model[H_id]
                new_H_chain = Chain(H_id)
                H_residues = [res.copy() for res in H_chain.get_residues() if res.id[1] <= H_max and res.id[0] == ' ']
                if H_residues:
                    for res in H_residues:
                        new_H_chain.add(res)
                    new_model.add(new_H_chain)
                    chains_added_this_run.add(H_id)
                    logging.info(f"PDB: {pdb_code} - added and truncated H chain {H_id} to new file.")
                else:
                    logging.warning(f"PDB: {pdb_code} - H chain {H_id} found in original file but has no residues after truncation.")
            elif H_id:
                logging.warning(f"PDB: {pdb_code} - H chain {H_id} not found in original file {pdb_path}.")

            if L_id and L_id in original_model:
                L_chain = original_model[L_id]
                new_L_chain = Chain(L_id)
                L_residues = [res.copy() for res in L_chain.get_residues() if res.id[1] <= L_max and res.id[0] == ' ']
                if L_residues:
                    for res in L_residues:
                        new_L_chain.add(res)
                    new_model.add(new_L_chain)
                    chains_added_this_run.add(L_id)
                    logging.info(f"PDB: {pdb_code} - added and truncated L chain {L_id} to new file.")
                else:
                    logging.warning(f"PDB: {pdb_code} - L chain {L_id} found in original file but has no residues after truncation.")
            elif L_id:
                logging.warning(f"PDB: {pdb_code} - L chain {L_id} not found in original file {pdb_path}.")

            for ag_chain_id in antigen_ids:
                if ag_chain_id in original_model:
                    ag_chain = original_model[ag_chain_id]
                    new_ag_chain = Chain(ag_chain_id)
                    ag_residues = [res.copy() for res in ag_chain.get_residues() if res.id[0] == ' ']
                    if ag_residues:
                        for res in ag_residues:
                            new_ag_chain.add(res)
                        new_model.add(new_ag_chain)
                        chains_added_this_run.add(ag_chain_id)
                        logging.info(f"PDB: {pdb_code} - added full Ag chain {ag_chain_id} to new file.")
                    else:
                         logging.warning(f"PDB: {pdb_code} - Ag chain {ag_chain_id} found in original file but has no standard residues.")
                else:
                    logging.warning(f"PDB: {pdb_code} - Ag chain {ag_chain_id} not found in original file {pdb_path}.")

            if chains_added_this_run:
                io.set_structure(new_structure)
                io.save(pdb_path_cut)
                logging.info(f"PDB: {pdb_code} - successfully created and saved to: {pdb_path_cut}, chains: {chains_added_this_run}")
            else:
                logging.warning(f"PDB: {pdb_code}, Entry: {entry.get('id', 'N/A')} - no chains added from original file to {pdb_path_cut}. File not created.")

        else:
            # --- Case 2: Output file DOES exist ---
            logging.info(f"PDB: {pdb_code}, Entry: {entry.get('id', 'N/A')} - target file {pdb_path_cut} already exists, checking whether new chains need to be added.")

            try:
                existing_structure = parser.get_structure('existing', pdb_path_cut)
                existing_model = existing_structure[0]
                existing_chain_ids = {chain.id for chain in existing_model}
            except (PDBExceptions.PDBException, FileNotFoundError, IndexError) as e:
                 logging.error(f"PDB: {pdb_code} - could not load or parse existing {pdb_path_cut}: {e}. Will try to overwrite.")
                 try:
                     os.remove(pdb_path_cut)
                 except OSError:
                     pass
                 return cut_antibody(task)


            if original_structure is None:
                original_structure = parser.get_structure('original', pdb_path)
                original_model = original_structure[0]

            added_new_chain = False
            chains_added_this_run = set()

            if H_id and H_id not in existing_chain_ids:
                if H_id in original_model:
                    H_chain = original_model[H_id]
                    new_H_chain = Chain(H_id)
                    H_residues = [res.copy() for res in H_chain.get_residues() if res.id[1] <= H_max and res.id[0] == ' ']
                    if H_residues:
                        for res in H_residues:
                            new_H_chain.add(res)
                        existing_model.add(new_H_chain)
                        added_new_chain = True
                        chains_added_this_run.add(H_id)
                        logging.info(f"PDB: {pdb_code} - added and truncated H chain {H_id} to existing file {pdb_path_cut}.")
                    else:
                        logging.warning(f"PDB: {pdb_code} - H chain {H_id} found in original file but has no residues after truncation, not added to {pdb_path_cut}.")
                else:
                    logging.warning(f"PDB: {pdb_code} - H chain {H_id} not found in original file {pdb_path}, cannot add to {pdb_path_cut}.")
            elif H_id:
                 logging.info(f"PDB: {pdb_code} - H chain {H_id} already present in {pdb_path_cut}.")


            if L_id and L_id not in existing_chain_ids:
                if L_id in original_model:
                    L_chain = original_model[L_id]
                    new_L_chain = Chain(L_id)
                    L_residues = [res.copy() for res in L_chain.get_residues() if res.id[1] <= L_max and res.id[0] == ' ']
                    if L_residues:
                        for res in L_residues:
                            new_L_chain.add(res)
                        existing_model.add(new_L_chain)
                        added_new_chain = True
                        chains_added_this_run.add(L_id)
                        logging.info(f"PDB: {pdb_code} - added and truncated L chain {L_id} to existing file {pdb_path_cut}.")
                    else:
                        logging.warning(f"PDB: {pdb_code} - L chain {L_id} found in original file but has no residues after truncation, not added to {pdb_path_cut}.")
                else:
                    logging.warning(f"PDB: {pdb_code} - L chain {L_id} not found in original file {pdb_path}, cannot add to {pdb_path_cut}.")
            elif L_id:
                 logging.info(f"PDB: {pdb_code} - L chain {L_id} already present in {pdb_path_cut}.")

            for ag_chain_id in antigen_ids:
                if ag_chain_id not in existing_chain_ids:
                    if ag_chain_id in original_model:
                        ag_chain = original_model[ag_chain_id]
                        new_ag_chain = Chain(ag_chain_id)
                        ag_residues = [res.copy() for res in ag_chain.get_residues() if res.id[0] == ' ']
                        if ag_residues:
                            for res in ag_residues:
                                new_ag_chain.add(res)
                            existing_model.add(new_ag_chain)
                            added_new_chain = True
                            chains_added_this_run.add(ag_chain_id)
                            logging.info(f"PDB: {pdb_code} - added full Ag chain {ag_chain_id} to existing file {pdb_path_cut}.")
                        else:
                            logging.warning(f"PDB: {pdb_code} - Ag chain {ag_chain_id} found in original file but has no standard residues, not added to {pdb_path_cut}.")
                    else:
                        logging.warning(f"PDB: {pdb_code} - Ag chain {ag_chain_id} not found in original file {pdb_path}, cannot add to {pdb_path_cut}.")
                else:
                    logging.info(f"PDB: {pdb_code} - Ag chain {ag_chain_id} already present in {pdb_path_cut}.")

            if added_new_chain:
                io.set_structure(existing_structure)
                io.save(pdb_path_cut)
                logging.info(f"PDB: {pdb_code} - successfully updated file: {pdb_path_cut}, newly added chains: {chains_added_this_run}")
            else:
                logging.info(f"PDB: {pdb_code}, Entry: {entry.get('id', 'N/A')} - no update needed for {pdb_path_cut}; all requested chains already present or invalid.")

        return pdb_path_cut

    except FileNotFoundError as e:
        logging.error(f"PDB: {pdb_code} - original PDB file not found: {pdb_path}, error: {str(e)}")
        raise e
    except PDBExceptions.PDBException as e:
        logging.error(f"PDB: {pdb_code} - error parsing PDB file ({pdb_path} or {pdb_path_cut}): {str(e)}")
        #     try: os.remove(pdb_path_cut) except OSError: pass
        raise e
    except Exception as e:
        logging.error(f"PDB: {pdb_code} - unexpected error while processing {pdb_path}: {str(e)}")
        raise e


import enum

class CDR(enum.IntEnum):
    H1 = 1
    H2 = 2
    H3 = 3
    L1 = 4
    L2 = 5
    L3 = 6


class IMGTCDRRange:
    H1 = (27, 38)
    H2 = (56, 65)
    H3 = (105, 117)

    L1 = (27, 38)
    L2 = (56, 65)
    L3 = (105, 117)





def preprocess_sabdab_structure_complex(task, interface = False, pad_coordinates=False):
    entry = task['entry']
    pdb_path = task['pdb_path']
    H_id = entry['H_chain']
    L_id = entry['L_chain']
    try:
        complex = ProteinComplex.from_pdb(pdb_path)
    except Exception as e:
        logging.warning(f"Failed to load PDB: {pdb_path}")
        return None
    metadata = complex.metadata
    residue_index = complex.residue_index
    chain_lookup = {v : k for k, v in metadata.chain_lookup.items()}
    Hchain = chain_lookup[H_id] if H_id is not None else None
    Lchain = chain_lookup[L_id] if L_id is not None else None
    chain_ids = complex.chain_id
    chain_ids = torch.tensor(chain_ids, dtype=torch.long)
    sequence = complex.sequence
    seq_tokens = tokenizer(sequence, truncation=True, padding='max_length', max_length=1024, return_tensors="pt")

    coordinates,plddt, index = to_structure_encoder_inputs(complex, should_normalize_coordinates=True)
    coordinates = coordinates.cuda()  # Ensure coords are on the same device as the encoder
    plddt = plddt.cuda()
    index = index.cuda()
    _, structure_tokens = encoder.encode(coordinates, residue_index=index)
    combined_structure_tokens = F.pad(structure_tokens, (1, 1), value=0)
    combined_structure_tokens[:, 0] = 4098
    combined_structure_tokens[:, -1] = 4097
    # Truncate or pad combined_structure_tokens to match combined_input_ids length
    combined_structure_tokens = combined_structure_tokens[:, :1024]
    chain_ids = F.pad(chain_ids, (1, 1), value=-2)
    # chain_ids[ 0] = -1
    # chain_ids[-1] = -1
    chain_ids = chain_ids[:1024]
    residue_index = F.pad(index, (1, 1), value=-1)
    residue_index = residue_index[:1024]
    L = combined_structure_tokens.size(1)
    padding_length = 1024 - L
    if padding_length > 0:
        combined_structure_tokens = F.pad(combined_structure_tokens, (0, padding_length), value=4099)
        chain_ids = F.pad(chain_ids, (0, padding_length), value=-2)
        residue_index = F.pad(residue_index, (0, padding_length), value=-1)

    cdr_pos = torch.zeros(1024)
    H_id_expanded = torch.full((1024,), Hchain)
    L_id_expanded = torch.full((1024,), Lchain)
    H_pos = (chain_ids == H_id_expanded).cuda()
    L_pos = (chain_ids == L_id_expanded).cuda()

    full_residue_index = residue_index.squeeze(0).cuda()

    cdr_pos[(full_residue_index >= 27) & (full_residue_index <= 38) & H_pos] = 1  # CDRH1
    cdr_pos[(full_residue_index >= 56) & (full_residue_index <= 65) & H_pos] = 2  # CDRH2
    cdr_pos[(full_residue_index >= 105) & (full_residue_index <= 117) & H_pos] = 3 # CDRH3

    cdr_pos[(full_residue_index >= 27) & (full_residue_index <= 38) & L_pos] = 4  # CDRL1
    cdr_pos[(full_residue_index >= 56) & (full_residue_index <= 65) & L_pos] = 5  # CDRL2
    cdr_pos[(full_residue_index >= 105) & (full_residue_index <= 117) & L_pos] = 6  # CDRL3

    if pad_coordinates:
        padded_coordinates = torch.zeros((1024, 37, 3), device=coordinates.device)
        coords_len = coordinates.shape[0]
        if coords_len > 1024:
            padded_coordinates = coordinates[:1024]
        else:
            padded_coordinates[:coords_len] = coordinates
        coordinates = padded_coordinates



    if interface:
        interface = torch.zeros_like(seq_tokens['input_ids'])
        # Get chain information
        ag_chains = []
        if entry['ag_chains'] is None:
            pass
        else:
            for ag_id in entry['ag_chains']:
                if ag_id in chain_lookup:
                    ag_chains.append(chain_lookup[ag_id])
            ab_chains = [Hchain, Lchain]
            ab_chains = [chain for chain in ab_chains if chain is not None]

            # Initialize coordinate dictionary
            coord_dict = {}
            chain_res_to_seq_map = {}
            current_seq_idx = 1

            for chain_idx in range(len(complex.atom37_positions)):
                chain_id = complex.chain_id[chain_idx]
                if chain_id in ag_chains + ab_chains:
                    if chain_id not in coord_dict:
                        coord_dict[chain_id] = {
                            'coordinates': [],
                            'residue_indices': [],
                            'seq_indices': []
                        }
                    # Use CA atom coordinates (index 1 in atom37)
                    ca_coord = complex.atom37_positions[chain_idx][1]
                    if not np.isnan(ca_coord).any():  # Check if CA exists
                        coord_dict[chain_id]['coordinates'].append(ca_coord)
                        coord_dict[chain_id]['residue_indices'].append(chain_idx)
                        coord_dict[chain_id]['seq_indices'].append(current_seq_idx)
                        chain_res_to_seq_map[(chain_id, chain_idx)] = current_seq_idx
                        current_seq_idx += 1

            # Convert coordinates to numpy arrays
            for chain_id in coord_dict:
                coord_dict[chain_id]['coordinates'] = np.array(coord_dict[chain_id]['coordinates'])

            # Calculate minimum distances between antigen and antibody residues
            interface_residues = set()
            distance_threshold = 10.0  # Angstroms

            for ag_chain in ag_chains:
                if ag_chain not in coord_dict:
                    continue
                ag_coords = coord_dict[ag_chain]['coordinates']

                min_distances_to_ab = np.inf * np.ones(len(ag_coords))

                for ab_chain in ab_chains:
                    if ab_chain not in coord_dict:
                        continue
                    ab_coords = coord_dict[ab_chain]['coordinates']
                    if len(ab_coords) == 0:
                        continue

                    # Calculate pairwise distances
                    dist_matrix = np.linalg.norm(ag_coords[:, None, :] - ab_coords[None, :, :], axis=-1)
                    chain_min_distances = np.min(dist_matrix, axis=1)

                    # Update minimum distances
                    min_distances_to_ab = np.minimum(min_distances_to_ab, chain_min_distances)


                # # Convert residue indices to sequence indices and add to interface set
                #     if seq_idx < len(interface[0]):  # Ensure index is within bounds

                if len(min_distances_to_ab) > 0:

                    sorted_indices = np.argsort(min_distances_to_ab)
                    num_interface = min(48, len(sorted_indices))
                    interface_res_indices = sorted_indices[:num_interface]

                    for idx in interface_res_indices:
                        orig_idx = coord_dict[ag_chain]['residue_indices'][idx]
                        seq_idx = coord_dict[ag_chain]['seq_indices'][idx]
                        if seq_idx < len(interface[0]):
                            interface[0][seq_idx] = 1
                            interface_residues.add(seq_idx)

    encoding = {
                'input_ids': seq_tokens['input_ids'].cpu(),
                'structure_tokens': combined_structure_tokens.cpu(),
                'chain_id': chain_ids.cpu(),
                'H_chain': Hchain,
                'L_chain': Lchain,
                'id': entry['id'],
                'cdr_pos': cdr_pos.cpu(),
                # 'interface': interface.cpu(),
                # 'coordinates': coordinates.cpu()
            }


    return encoding

def preprocess_sabdab_structure_single_chain(task, interface = False, pad_coordinates=False):
    entry = task['entry']
    pdb_path = task['pdb_path']
    H_id = entry['H_chain']
    L_id = entry['L_chain']
    ag_ids = entry.get('ag_chains', [])
    ag_chains = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        H_chain = ProteinChain.from_pdb(pdb_path, H_id)
        L_chain = ProteinChain.from_pdb(pdb_path, L_id)

        valid_ag_ids = []
        for ag_id in ag_ids:
            try:
                ag_chain = ProteinChain.from_pdb(pdb_path, ag_id)
                ag_chains.append(ag_chain)
                valid_ag_ids.append(ag_id)
            except Exception as e:
                logging.warning(f"Failed to load antigen chain {ag_id}: {e}")

        if ag_ids and not ag_chains:
            logging.warning(f"Failed to load any antigen chain: {ag_ids}")
    except Exception as e:
        logging.warning(f"Failed to load PDB file: {pdb_path}, error: {e}")
        return None

    H_sequence = H_chain.sequence
    L_sequence = L_chain.sequence
    ag_sequences = [chain.sequence for chain in ag_chains]
    ag_seq = ''.join(ag_sequences)

    if len(ag_seq) > 800:
        logging.warning(f"Antigen sequence too long ({len(ag_seq)} > 800): {pdb_path}")
        return None

    H_tokens = tokenizer(H_sequence, truncation=True, return_tensors="pt")
    L_tokens = tokenizer(L_sequence, truncation=True, return_tensors="pt")
    ag_tokens = [tokenizer(ag_sequence, truncation=True, return_tensors="pt") for ag_sequence in ag_sequences]

    complex_tokens = tokenizer(H_sequence + '|' + L_sequence + '|' + '|'.join(ag_sequences),
                               truncation=True, padding='max_length', max_length=1024, return_tensors="pt")

    try:
        H_coordinates, H_plddt, H_index = to_structure_encoder_inputs(H_chain, should_normalize_coordinates=True)
        L_coordinates, L_plddt, L_index = to_structure_encoder_inputs(L_chain, should_normalize_coordinates=True)


        ag_coordinates = []
        ag_index = []
        for i, ag_chain in enumerate(ag_chains):
            try:
                ag_coord, _, ag_idx = to_structure_encoder_inputs(ag_chain, should_normalize_coordinates=True)
                ag_coordinates.append(ag_coord)
                ag_index.append(ag_idx)
            except Exception as e:
                logging.warning(f"Error processing antigen chain {valid_ag_ids[i]}: {e}")

        if ag_ids and not ag_coordinates:
            logging.warning(f"All antigen chain coordinates are invalid: {valid_ag_ids}")
    except Exception as e:
        logging.warning(f"Error getting coordinates: {e}")
        return None

    try:
        H_coordinates = H_coordinates.to(device)
        L_coordinates = L_coordinates.to(device)
        ag_coordinates = [ag_coord.to(device) for ag_coord in ag_coordinates]
        H_index = H_index.to(device)
        L_index = L_index.to(device)
        ag_index = [ag_idx.to(device) for ag_idx in ag_index]
    except Exception as e:
        logging.error(f"Error moving tensors to device: {e}")
        device = torch.device('cpu')
        try:
            logging.info("Trying to process on CPU")
            H_coordinates = H_coordinates.to(device)
            L_coordinates = L_coordinates.to(device)
            ag_coordinates = [ag_coord.to(device) for ag_coord in ag_coordinates]
            H_index = H_index.to(device)
            L_index = L_index.to(device)
            ag_index = [ag_idx.to(device) for ag_idx in ag_index]
        except Exception as e2:
            logging.error(f"Processing on CPU also failed: {e2}")
            return None

    try:
        stru_separator = torch.full((1, 1), 4100, device=device)
        _, H_structure_tokens = encoder.encode(H_coordinates, residue_index=H_index)
        _, L_structure_tokens = encoder.encode(L_coordinates, residue_index=L_index)
    except Exception as e:
        logging.error(f"Error encoding antibody structure: {e}")
        return None

    try:
        ag_structure_tokens = []
        for ag_coord, ag_idx in zip(ag_coordinates, ag_index):
            try:
                _, ag_token = encoder.encode(ag_coord, residue_index=ag_idx)
                ag_structure_tokens.append(ag_token)
            except Exception as e:
                logging.warning(f"Error encoding antigen structure, skipping this antigen: {e}")
    except Exception as e:
        logging.error(f"Error processing antigen structure tokens: {e}")
        ag_structure_tokens = []

    ag_tokens_with_separator = []
    for ag_token in ag_structure_tokens:
        ag_tokens_with_separator.append(ag_token)
        ag_tokens_with_separator.append(stru_separator)

    coordinates_separator = torch.full((1, 1, 37, 3), float('nan'), device=device)

    try:
        if ag_coordinates:
            combined_coordinates = torch.cat([H_coordinates, L_coordinates] + ag_coordinates, dim=1)
        else:
            combined_coordinates = torch.cat([H_coordinates, L_coordinates], dim=1)
    except Exception as e:
        logging.error(f"Error combining coordinates: {e}")
        return None

    ag_coordinates_with_separator = []
    for ag_coord in ag_coordinates:
        ag_coordinates_with_separator.append(ag_coord)
        ag_coordinates_with_separator.append(coordinates_separator)

    try:
        if ag_coordinates_with_separator:
            combined_coordinates_with_sep = torch.cat(
                [coordinates_separator, H_coordinates, coordinates_separator,
                 L_coordinates, coordinates_separator] + ag_coordinates_with_separator, dim=1)
        else:
            combined_coordinates_with_sep = torch.cat(
                [coordinates_separator, H_coordinates, coordinates_separator,
                 L_coordinates, coordinates_separator], dim=1)
    except Exception as e:
        logging.error(f"Error combining coordinates with separators: {e}")
        return None

    try:
        if ag_tokens_with_separator:
            structure_tokens = torch.cat(
                [H_structure_tokens, stru_separator, L_structure_tokens, stru_separator] + ag_tokens_with_separator, dim=1)
        else:
            structure_tokens = torch.cat(
                [H_structure_tokens, stru_separator, L_structure_tokens, stru_separator], dim=1)
    except Exception as e:
        logging.error(f"Error combining structure tokens: {e}")
        return None

    # Truncate or pad structure_tokens to match input_ids length
    structure_tokens = structure_tokens[:, :1024]
    combined_structure_tokens = F.pad(structure_tokens, (1, 1), value=0)
    combined_structure_tokens[:, 0] = 4098
    combined_structure_tokens[:, -1] = 4097

    # Create chain_id_separator and chain IDs
    try:
        chain_id_separator = torch.full((1,1), -1, device=device)
        H_chain_ids = torch.full((H_structure_tokens.size(0), H_structure_tokens.size(1)), 0, device=device)
        L_chain_ids = torch.full((L_structure_tokens.size(0), L_structure_tokens.size(1)), 1, device=device)

        ag_chain_ids = []
        for i, ag_token in enumerate(ag_structure_tokens):
            ag_chain_ids.append(torch.full((ag_token.size(0), ag_token.size(1)), i + 2, device=device))

        ag_chain_ids_with_separator = []
        for ag_chain_id in ag_chain_ids:
            ag_chain_ids_with_separator.append(ag_chain_id)
            ag_chain_ids_with_separator.append(chain_id_separator)

        if ag_chain_ids_with_separator:
            chain_ids = torch.cat([H_chain_ids, chain_id_separator, L_chain_ids, chain_id_separator] +
                                ag_chain_ids_with_separator, dim=1)
        else:
            chain_ids = torch.cat([H_chain_ids, chain_id_separator, L_chain_ids, chain_id_separator], dim=1)

        chain_ids = F.pad(chain_ids, (1, 1), value=-2)
        chain_ids = chain_ids[:1024]
    except Exception as e:
        logging.error(f"Error processing chain IDs: {e}")
        return None

    # Create index_separator and process residue indices
    try:
        index_separator = torch.full((1,1), -1, device=device)

        ag_index_with_separator = []
        for ag_idx in ag_index:
            ag_index_with_separator.append(ag_idx)
            ag_index_with_separator.append(index_separator)

        if ag_index_with_separator:
            residue_index = torch.cat([H_index, index_separator, L_index, index_separator] +
                                   ag_index_with_separator, dim=1)
        else:
            residue_index = torch.cat([H_index, index_separator, L_index, index_separator], dim=1)

        residue_index = F.pad(residue_index, (1, 1), value=-1)
        residue_index = residue_index[:1024]

        L = combined_structure_tokens.size(1)
        padding_length = 1024 - L
        if padding_length > 0:
            combined_structure_tokens = F.pad(combined_structure_tokens, (0, padding_length), value=4099)
            chain_ids = F.pad(chain_ids, (0, padding_length), value=-2)
            residue_index = F.pad(residue_index, (0, padding_length), value=-1)
    except Exception as e:
        logging.error(f"Error processing residue indices: {e}")
        return None

    # Process CDR positions
    try:
        cdr_pos = torch.zeros(1024, device=device)
        H_id_expanded = torch.full((1024,), 0, device=device)
        L_id_expanded = torch.full((1024,), 1, device=device)
        H_pos = (chain_ids == H_id_expanded).squeeze(0)
        L_pos = (chain_ids == L_id_expanded).squeeze(0)

        full_residue_index = residue_index.squeeze(0)

        cdr_pos[(full_residue_index >= 27) & (full_residue_index <= 38) & H_pos] = 1  # CDRH1
        cdr_pos[(full_residue_index >= 56) & (full_residue_index <= 65) & H_pos] = 2  # CDRH2
        cdr_pos[(full_residue_index >= 105) & (full_residue_index <= 117) & H_pos] = 3 # CDRH3

        cdr_pos[(full_residue_index >= 27) & (full_residue_index <= 38) & L_pos] = 4  # CDRL1
        cdr_pos[(full_residue_index >= 56) & (full_residue_index <= 65) & L_pos] = 5  # CDRL2
        cdr_pos[(full_residue_index >= 105) & (full_residue_index <= 117) & L_pos] = 6  # CDRL3
    except Exception as e:
        logging.error(f"Error processing CDR positions: {e}")
        return None

    # Pad coordinates if needed
    if pad_coordinates:
        try:
            padded_coordinates = torch.zeros((1024, 37, 3), device=device)
            coords_len = combined_coordinates.shape[1]
            if coords_len > 1024:
                padded_coordinates = combined_coordinates[:, :1024, :, :]
            else:
                padded_coordinates[:, :coords_len, :, :] = combined_coordinates
            combined_coordinates = padded_coordinates
        except Exception as e:
            logging.error(f"Error padding coordinates: {e}")

    # Process interface if needed
    if interface:
        try:
            interface = torch.zeros_like(complex_tokens['input_ids'])

            if entry['ag_chains'] is not None:
                coord_dict = {}
                chain_res_to_seq_map = {}
                current_seq_idx = 1

                coord_dict[0] = {
                    'coordinates': [],
                    'residue_indices': [],
                    'seq_indices': []
                }
                for idx in range(len(H_chain.atom37_positions)):
                    ca_coord = H_chain.atom37_positions[idx][1]
                    if not np.isnan(ca_coord).any():
                        coord_dict[0]['coordinates'].append(ca_coord)
                        coord_dict[0]['residue_indices'].append(idx)
                        coord_dict[0]['seq_indices'].append(current_seq_idx)
                        chain_res_to_seq_map[(0, idx)] = current_seq_idx
                        current_seq_idx += 1

                coord_dict[1] = {
                    'coordinates': [],
                    'residue_indices': [],
                    'seq_indices': []
                }
                for idx in range(len(L_chain.atom37_positions)):
                    ca_coord = L_chain.atom37_positions[idx][1]
                    if not np.isnan(ca_coord).any():
                        coord_dict[1]['coordinates'].append(ca_coord)
                        coord_dict[1]['residue_indices'].append(idx)
                        coord_dict[1]['seq_indices'].append(current_seq_idx)
                        chain_res_to_seq_map[(1, idx)] = current_seq_idx
                        current_seq_idx += 1

                for chain_idx, ag_chain in enumerate(ag_chains):
                    ag_chain_id = chain_idx + 2
                    coord_dict[ag_chain_id] = {
                        'coordinates': [],
                        'residue_indices': [],
                        'seq_indices': []
                    }
                    for idx in range(len(ag_chain.atom37_positions)):
                        ca_coord = ag_chain.atom37_positions[idx][1]
                        if not np.isnan(ca_coord).any():
                            coord_dict[ag_chain_id]['coordinates'].append(ca_coord)
                            coord_dict[ag_chain_id]['residue_indices'].append(idx)
                            coord_dict[ag_chain_id]['seq_indices'].append(current_seq_idx)
                            chain_res_to_seq_map[(ag_chain_id, idx)] = current_seq_idx
                            current_seq_idx += 1

                for chain_id in coord_dict:
                    coord_dict[chain_id]['coordinates'] = np.array(coord_dict[chain_id]['coordinates'])

                distance_threshold = 10.0
            all_ag_distances = []

            for ag_chain_id in range(2, 2 + len(ag_chains)):
                if ag_chain_id not in coord_dict:
                    continue
                ag_coords = coord_dict[ag_chain_id]['coordinates']

                min_distances_to_ab = np.inf * np.ones(len(ag_coords))

                for ab_chain_id in [0, 1]:
                    if ab_chain_id not in coord_dict:
                        continue
                    ab_coords = coord_dict[ab_chain_id]['coordinates']
                    if len(ab_coords) == 0:
                        continue

                    dist_matrix = np.linalg.norm(ag_coords[:, None, :] - ab_coords[None, :, :], axis=-1)
                    chain_min_distances = np.min(dist_matrix, axis=1)

                    min_distances_to_ab = np.minimum(min_distances_to_ab, chain_min_distances)

                for i in range(len(min_distances_to_ab)):
                    all_ag_distances.append({
                        'distance': min_distances_to_ab[i],
                        'chain_id': ag_chain_id,
                        'local_idx': i
                    })

            all_ag_distances.sort(key=lambda x: x['distance'])

            interface_residues = set()
            orig_idx_interface_residues = set()
            num_interface = min(64, len(all_ag_distances))

            for i in range(num_interface):
                res_info = all_ag_distances[i]
                chain_id = res_info['chain_id']
                local_idx = res_info['local_idx']

                orig_idx = coord_dict[chain_id]['residue_indices'][local_idx]
                seq_idx = coord_dict[chain_id]['seq_indices'][local_idx]

                if seq_idx < len(interface[0]):
                    interface[0][seq_idx] = 1
                    interface_residues.add(seq_idx)
                    orig_idx_interface_residues.add(orig_idx)



        except Exception as e:
            logging.error(f"Error processing interface: {e}")

    # Prepare final encoding
    try:
        encoding = {
            'input_ids': complex_tokens['input_ids'].cpu(),
            'structure_tokens': combined_structure_tokens.cpu(),
            'chain_id': chain_ids.cpu(),
            'H_chain': 0,
            'L_chain': 1,
            'id': entry['id'],
            'cdr_pos': cdr_pos.cpu(),
            'coordinates':combined_coordinates.cpu(),
            'interface':interface.cpu()
        }



        return encoding
    except Exception as e:
        logging.error(f"Error creating final encoding: {e}")
        return None


def _aa_tensor_to_sequence(aa):
    return ''.join([Polypeptide.index_to_one(a.item()) for a in aa.flatten()])

class ProteinDataset_antigen(Dataset):
    def __init__(self, max_length=1024,
                summary_path = 'datasets/SFT/summary/sabdab_summary_all.tsv',
                imgt_dir = 'datasets/SFT/imgt',
                processed_dir = 'datasets/SFT/imgt_processed_iggm',
                init = False, split = 'train'
        ):

        self.max_length = max_length
        self.summary_path = summary_path
        self.imgt_dir = imgt_dir
        self.processed_dir = processed_dir
        self._load_sabdab_entries()
        if init:
            self._preprocess_structures()
        self._load_clusters()
        self.load_split(split)
        output_file = os.path.join(self.processed_dir, 'processed_single_epitope.npz')
        self.loaded_data = np.load(output_file, allow_pickle=True)['data']

    def __len__(self):
        return len(self.ids_in_split)

    def to_structure_encoder_inputs(
        self,complex,
        should_normalize_coordinates: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coords = torch.tensor(complex.atom37_positions, dtype=torch.float32)
        plddt = torch.tensor(complex.confidence, dtype=torch.float32)
        residue_index = torch.tensor(complex.residue_index, dtype=torch.long)

        if should_normalize_coordinates:
            coords = normalize_coordinates(coords)
        return coords.unsqueeze(0), plddt.unsqueeze(0), residue_index.unsqueeze(0)


    def _load_sabdab_entries(self):
        df = pd.read_csv(self.summary_path, sep='\t')
        entries_all = []
        for i, row in tqdm(
            df.iterrows(),
            dynamic_ncols=True,
            desc='Loading entries',
            total=len(df),
        ):
            entry_id = "{pdbcode}_{H}_{L}_{Ag}".format(
                pdbcode = row['pdb'],
                H = nan_to_empty_string(row['Hchain']),
                L = nan_to_empty_string(row['Lchain']),
                Ag = ''.join(split_sabdab_delimited_str(
                    nan_to_empty_string(row['antigen_chain'])
                ))
            )
            ag_chains = split_sabdab_delimited_str(
                nan_to_empty_string(row['antigen_chain'])
            )
            resolution = parse_sabdab_resolution(row['resolution'])
            entry = {
                'id': entry_id,
                'pdbcode': row['pdb'],
                'H_chain': nan_to_none(row['Hchain']),
                'L_chain': nan_to_none(row['Lchain']),
                'ag_chains': ag_chains,
                'ag_type': nan_to_none(row['antigen_type']),
                'ag_name': nan_to_none(row['antigen_name']),
                'date': datetime.datetime.strptime(row['date'], '%m/%d/%y'),
                'resolution': resolution,
                'method': row['method'],
                'scfv': row['scfv'],
            }

            # Filtering
            if (
                (entry['ag_type'] in ALLOWED_AG_TYPES or entry['ag_type'] is None)
                and (entry['resolution'] is not None and entry['resolution'] <= RESOLUTION_THRESHOLD)
            ):
                entries_all.append(entry)
        self.sabdab_entries = entries_all

    def _preprocess_structures(self):
        tasks = []
        for entry in self.sabdab_entries:
            pdb_path = os.path.join(self.imgt_dir, '{}.pdb'.format(entry['pdbcode']))
            if not os.path.exists(pdb_path):
                logging.warning(f"PDB not found: {pdb_path}")
                continue

            tasks.append({
                'id': entry['id'],
                'entry': entry,
                'pdb_path': pdb_path,
            })

        data_list = []
        for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocess'):
            try:
                pdb_path = cut_antibody(task)
                task['pdb_path'] = pdb_path
                data = preprocess_sabdab_structure_single_chain(task, interface=True)
            except Exception as e:
                print(e)
                data = None
            if data is not None:
                data_list.append(data)
        processed_list = []
        for data in tqdm(data_list):
            if data is not None:
                processed_data = {
                    'input_ids': data['input_ids'].numpy(),
                    'coordinates': data['coordinates'].numpy(),
                # 'plddt': item['plddt'].numpy(),
                    'chain_id': data['chain_id'].numpy(),
                    'H_chain': data['H_chain'],
                    'L_chain': data['L_chain'],
                    'structure_tokens': data['structure_tokens'].numpy(),
                    'id':data['id'],
                    'cdr_pos': data['cdr_pos'].numpy(),
                    'interface': data['interface'].numpy()
                }
                processed_list.append(processed_data)

        npz_path = os.path.join(self.processed_dir, 'processed_single_epitope.npz')
        np.savez_compressed(npz_path, data=processed_list)

    @property
    def _cluster_path(self):
        return os.path.join(self.processed_dir, 'cluster_result_cluster.tsv')

    def _load_clusters(self):

        clusters, id_to_cluster = {}, {}
        with open(self._cluster_path, 'r') as f:
            for line in f.readlines():
                cluster_name, data_id = line.split()
                if cluster_name not in clusters:
                    clusters[cluster_name] = []
                clusters[cluster_name].append(data_id)
                id_to_cluster[data_id] = cluster_name
        self.clusters = clusters
        self.id_to_cluster = id_to_cluster

    def load_split(self, split):
        assert split in ('train', 'val', 'test')
        ids_test = [
            entry['id']
            for entry in self.sabdab_entries
            if entry['pdbcode'] in IGGM_IDS
            # if entry['pdbcode'] in RAbD_IDS
        ]
        test_relevant_clusters = set()
        for id in ids_test:
            try:
                test_relevant_clusters.add(self.id_to_cluster[id])
            except Exception as e:
                a=1
        #     # print(e)

        ids_train_val = []

        for entry in self.sabdab_entries:
            try:
                if self.id_to_cluster[entry['id']] not in test_relevant_clusters:
                    ids_train_val.append(entry['id'])
            except Exception as e:
                b = 1


        random.Random(42).shuffle(ids_train_val)
        if split == 'test':
            self.ids_in_split = ids_test
        elif split == 'val':
            self.ids_in_split = ids_train_val[:20]
        else:
            self.ids_in_split = ids_train_val[20:]


    def get_structure(self,id):


        for item in self.loaded_data:
            if item['id'] == id:
                return item

        raise ValueError(f"Structure with id {id} not found in the dataset")
    def __getitem__(self, index):
        id = self.ids_in_split[index]
        processed_item = {}
        try:
            item = self.get_structure(id)
        except ValueError as e:
            return self[random.randint(0, len(self) - 1)]
        item['coordinates'] = torch.zeros((1024, 37, 3))
        for k, v in item.items():
            if isinstance(v, np.ndarray):
                v = torch.tensor(v)
                v = v.squeeze()
            processed_item[k] = v
        return processed_item
class antibody_dataset(ProteinDataset_antigen):
    def __getitem__(self, index):
        item =  super().__getitem__(index)
        chain_id = item['chain_id'] #1024
        cdr_pos = item['cdr_pos']
        sequence_tokens = item['input_ids']
        structure_tokens = item['structure_tokens']

        antibody_mask = (chain_id == 0) | (chain_id == 1)

        ab_chain_id = chain_id[antibody_mask]
        ab_cdr_pos = cdr_pos[antibody_mask]
        ab_sequence_tokens = sequence_tokens[antibody_mask]
        ab_structure_tokens = structure_tokens[antibody_mask]

        ab_length = ab_chain_id.shape[0]

        padded_length = 512
        padded_chain_id = torch.full((padded_length,), -1, dtype=ab_chain_id.dtype)
        padded_cdr_pos = torch.zeros(padded_length, dtype=ab_cdr_pos.dtype)
        padded_sequence_tokens = torch.full((padded_length,), 1, dtype=ab_sequence_tokens.dtype)
        padded_structure_tokens = torch.full((padded_length,), 4099, dtype=ab_structure_tokens.dtype)

        if ab_length > padded_length:
            padded_chain_id = ab_chain_id[:padded_length]
            padded_cdr_pos = ab_cdr_pos[:padded_length]
            padded_sequence_tokens = ab_sequence_tokens[:padded_length]
            padded_structure_tokens = ab_structure_tokens[:padded_length]
        else:
            padded_chain_id[:ab_length] = ab_chain_id
            padded_cdr_pos[:ab_length] = ab_cdr_pos
            padded_sequence_tokens[:ab_length] = ab_sequence_tokens
            padded_structure_tokens[:ab_length] = ab_structure_tokens

        item['chain_id'] = padded_chain_id
        item['cdr_pos'] = padded_cdr_pos
        item['input_ids'] = padded_sequence_tokens
        item['structure_tokens'] = padded_structure_tokens

        return item


# print(len(dataset))


#     def __init__(self, max_length=1024,

#         ):





#         for i, row in tqdm(
#             df.iterrows(),
#         ):
#                 ))
#                 'id': entry_id,
#                 'pdbcode': row['pdb'],
#                 'H_chain': nan_to_none(row['Hchain']),
#                 'L_chain': nan_to_none(row['Lchain']),
#                 'ag_chains': ag_chains,
#                 'ag_type': nan_to_none(row['antigen_type']),
#                 'ag_name': nan_to_none(row['antigen_name']),
#                 'date': datetime.datetime.strptime(row['date'], '%m/%d/%y'),
#                 'resolution': resolution,
#                 'method': row['method'],
#                 'scfv': row['scfv'],

#             # Filtering
#             if (
#                 (entry['ag_type'] in ALLOWED_AG_TYPES or entry['ag_type'] is None)
#                 and (entry['resolution'] is not None and entry['resolution'] <= RESOLUTION_THRESHOLD)
#             ):


#             tasks.append({
#                 'id': entry['id'],
#                 'entry': entry,
#                 'pdb_path': pdb_path,
#             })
