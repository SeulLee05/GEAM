from time import time
from rdkit import Chem

import torch
from torch_geometric.loader import DataLoader

from utils_fgib.data import DataClass
from model.fgib import FGIB

import warnings
warnings.filterwarnings('ignore')


def get_loader(target, batch_size):
    start_time = time()
    train, test = torch.load('data/zinc250k.pt')
    print(f'{time() - start_time:.2f} sec for data loading')

    train, test = DataClass(train, target), DataClass(test, target)
    print(f'Train: {len(train)} | Test: {len(test)}')

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
    
    
def get_load_model(ckpt, device):
    state_dict = torch.load(ckpt, map_location=device)['state_dict']
    model = FGIB(device).to(device)
    model.load_state_dict(state_dict)
    return model


def get_sanitize_error_frags(frags):
    benzene = Chem.MolFromSmiles('c1ccccc1')
    att = Chem.MolFromSmarts('[#0]')

    error_frags = []
    for frag in frags:
        mols = Chem.ReplaceSubstructs(Chem.MolFromSmiles(frag), att, benzene)
        sanitize_error = False
        for mol in mols:
            mol = Chem.DeleteSubstructs(mol, att)
            try:
                Chem.SanitizeMol(mol)
            except:
                sanitize_error = True
        if sanitize_error:
            error_frags.append(frag)
    return set(error_frags)
