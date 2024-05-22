# Code adapted and modified from CGIB
import re
import json

import torch
from tqdm import tqdm
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torch_geometric.data import Data

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors as rdDesc
from rdkit.Chem import BRICS
RDLogger.DisableLog('rdApp.*')


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))


def get_atom_features(atom, stereo, features, explicit_H=False):
    """
    Method that computes atom level features from rdkit atom object
    :param atom:
    :param stereo:
    :param features:
    :param explicit_H:
    :return: the node features of an atom
    """
    possible_atoms = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    atom_features = one_of_k_encoding_unk(atom.GetSymbol(), possible_atoms)
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3])
    atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1])
    atom_features += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D])
    atom_features += [int(i) for i in list("{0:06b}".format(features))]

    if not explicit_H:
        atom_features += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    try:
        atom_features += one_of_k_encoding_unk(stereo, ['R', 'S'])
        atom_features += [atom.HasProp('_ChiralityPossible')]
    except Exception as e:
        atom_features += [False, False] + [atom.HasProp('_ChiralityPossible')]

    return np.array(atom_features)


def get_bond_features(bond):
    """
    Method that computes bond level features from rdkit bond object
    :param bond: rdkit bond object
    :return: bond features, 1d numpy array
    """
    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    bond_feats += one_of_k_encoding_unk(str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])

    return np.array(bond_feats)


def get_frag_batch_brics(mol, get_frag_only=False):
    [atom.SetAtomMapNum(i + 1) for i, atom in enumerate(mol.GetAtoms())]

    bonds = [bond[0] for bond in list(BRICS.FindBRICSBonds(mol))]
    bonds = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bonds]

    if not bonds:
        return None, None, None

    frags = Chem.MolToSmiles(Chem.FragmentOnBonds(mol, bonds)).split('.')
    frags = [Chem.MolFromSmiles(frag) for frag in frags]

    frag_idx = []
    for frag in frags:
        frag_idx.append([atom.GetAtomMapNum() - 1 for atom in frag.GetAtoms()
                         if atom.GetAtomMapNum() > 0])
        [atom.SetAtomMapNum(0) for atom in frag.GetAtoms()]

    frags = [Chem.MolToSmiles(frag) for frag in frags]
    frags = [re.sub(r'\[[0-9]+\*\]', '*', frag) for frag in frags]
    frags = [re.sub(r'\*', '[*:1]', frag) for frag in frags]
    frags = [Chem.MolToSmiles(Chem.MolFromSmiles(frag), isomericSmiles=False) for frag in frags]

    if get_frag_only:
        return frags
    
    node2frag_batch = torch.zeros(mol.GetNumAtoms(), dtype=torch.int64)
    for batch_id, idx_list in enumerate(frag_idx):
        node2frag_batch[idx_list] = batch_id
    
    frag2graph_batch = torch.zeros(len(frags), dtype=torch.int64)
    
    return frags, node2frag_batch, frag2graph_batch


def get_graph(smiles, idx=0):
    mol = Chem.MolFromSmiles(smiles)
    features = rdDesc.GetFeatureInvariants(mol)

    stereo = Chem.FindMolChiralCenters(mol)
    chiral_centers = [0] * mol.GetNumAtoms()
    for i in stereo:
        chiral_centers[i[0]] = i[1]

    node_features = []
    edge_features = []
    bonds = []
    for i in range(mol.GetNumAtoms()):
        atom_i = mol.GetAtomWithIdx(i)

        atom_i_features = get_atom_features(atom_i, chiral_centers[i], features[i])
        node_features.append(atom_i_features)

        for j in range(mol.GetNumAtoms()):
            bond_ij = mol.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                bonds.append([i, j])
                bond_features_ij = get_bond_features(bond_ij)
                edge_features.append(bond_features_ij)

    atom_feats = torch.tensor(np.array(node_features), dtype=torch.float)
    edge_index = torch.tensor(np.array(bonds), dtype=torch.long).T
    edge_feats = torch.tensor(np.array(edge_features), dtype=torch.float)

    frags, node2frag_batch, frag2graph_batch = get_frag_batch_brics(mol)

    if frags is not None:
        return Data(x=atom_feats, edge_index=edge_index, edge_attr=edge_feats,
                    idx=idx, smiles=Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False),
                    frags=frags, node2frag_batch=node2frag_batch, frag2graph_batch=frag2graph_batch)
    

def build_dataset(df):
    processed = []
    for idx in tqdm(range(len(df))):
        smiles = df.iloc[idx]['smiles']
        graph = get_graph(smiles, idx)
        if graph is not None:
            value = {'parp1': df.iloc[idx]['parp1'],
                     'fa7': df.iloc[idx]['fa7'],
                     '5ht1b': df.iloc[idx]['5ht1b'],
                     'braf': df.iloc[idx]['braf'],
                     'jak2': df.iloc[idx]['jak2'],
                     'qed': df.iloc[idx]['qed'],
                     'sa': df.iloc[idx]['sa'],
                     'amlodipine_mpo': df.iloc[idx]['amlodipine_mpo'],
                     'fexofenadine_mpo': df.iloc[idx]['fexofenadine_mpo'],
                     'osimertinib_mpo': df.iloc[idx]['osimertinib_mpo'],
                     'perindopril_mpo': df.iloc[idx]['perindopril_mpo'],
                     'ranolazine_mpo': df.iloc[idx]['ranolazine_mpo'],
                     'sitagliptin_mpo': df.iloc[idx]['sitagliptin_mpo'],
                     'zaleplon_mpo': df.iloc[idx]['zaleplon_mpo']}
            processed.append((graph, value))
    return processed


class DataClass(Dataset):
    def __init__(self, dataset, target):
        self.dataset = dataset
        self.target = target

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        graph, value = self.dataset[idx]
        if self.target in ['parp1', 'fa7', '5ht1b', 'braf', 'jak2']:
            return graph, np.clip(value[self.target], 0, 20) / 20 * value['qed'] * value['sa']
        else:   # mpo
            return graph, value[self.target]


if __name__ == "__main__":
    df = pd.read_csv('data/zinc250k.csv')
    with open('data/valid_idx_zinc250k.json') as f:
        test_idx = json.load(f)
    train_idx = [i for i in range(len(df)) if i not in test_idx]

    train = build_dataset(df.iloc[train_idx])
    test = build_dataset(df.iloc[test_idx])
    torch.save((train, test), 'data/zinc250k.pt')
    print(f'Train: {len(train)} | Test: {len(test)}')
