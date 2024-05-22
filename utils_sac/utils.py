import os
import sys
import copy
import numpy as np
import random
import torch
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, QED, RDConfig
from rdkit.Chem.rdmolops import FastFindRings
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def reward_vina(smis, predictor):
    reward = - np.array(predictor.predict(smis))
    reward = np.clip(reward, 0, None)
    return reward


def reward_qed(mols):
    return [QED.qed(m) for m in mols]


def reward_sa(mols):
    return [(10 - sascorer.calculateScore(m)) / 9 for m in mols]


def get_att_points(mol):
    att_points = []
    for a in mol.GetAtoms():
        if a.GetSymbol() == '*':
            att_points.append(a.GetIdx())
    return att_points


def delete_multiple_element(list_object, indices):
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(list_object):
            list_object.pop(idx)


def get_vocab(vocab_path):
    global ATOM
    
    ATOM = ['C', 'N', 'O', 'S', 'P', 'F', 'I', 'Cl','Br', '*']
    df = pd.read_csv(vocab_path, names=['frag', 'score'])
    FRAG = df['frag'].tolist()
    FRAG_QUEUE = list(zip(df['frag'], df['score']))
    FRAG_MOL = [Chem.MolFromSmiles(s) for s in FRAG]
    FRAG_ATT = [get_att_points(m) for m in FRAG_MOL]
    return {'ATOM': ATOM, 'FRAG': FRAG, 'FRAG_QUEUE': FRAG_QUEUE, 'FRAG_MOL': FRAG_MOL, 'FRAG_ATT': FRAG_ATT}


def ecfp(molecule):
    molecule = Chem.DeleteSubstructs(molecule, Chem.MolFromSmiles("*"))
    molecule.UpdatePropertyCache()
    FastFindRings(molecule)
    return [x for x in AllChem.GetMorganFingerprintAsBitVect(molecule, 2, 1024)]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: float(x == s), allowable_set))


def atom_feature(atom, use_atom_meta):
    if use_atom_meta == False:
        return np.asarray(
            one_of_k_encoding_unk(atom.GetSymbol(), ATOM) 
            )
    else:
        return np.asarray(
            one_of_k_encoding_unk(atom.GetSymbol(), ATOM) +
            one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
            one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
            one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
            [atom.GetIsAromatic()])


def convert_radical_electrons_to_hydrogens(mol):
    m = copy.deepcopy(mol)
    if Chem.Descriptors.NumRadicalElectrons(m) == 0:  # not a radical
        return m
    else:  # a radical
        for a in m.GetAtoms():
            num_radical_e = a.GetNumRadicalElectrons()
            if num_radical_e > 0:
                a.SetNumRadicalElectrons(0)
                a.SetNumExplicitHs(num_radical_e)
    return m
