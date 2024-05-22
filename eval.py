import os
import argparse
import pandas as pd
from rdkit import Chem
from utils_sac.utils import reward_qed, reward_sa
from utils_sac.utils_eval import get_novelty, get_ncircle


def get_hit_top5(df):
    hit_ratio = len(df[df['DOCKING'] > hit_thr]) / n_total_smi
    idx_tmp = int(n_total_smi * .05)
    top_5_score = df.sort_values(by='DOCKING', ascending=False)['DOCKING'].iloc[:idx_tmp].mean()
    return hit_ratio, top_5_score


parser = argparse.ArgumentParser()
parser.add_argument('file_path', type=str)
parser.add_argument('-t', '--target', type=str, default='parp1',
                    choices=['parp1', 'fa7', '5ht1b', 'braf', 'jak2'])
args = parser.parse_args()

if not os.path.exists('data/zinc250k_novelty.pt'):
    import json
    import torch
    from rdkit import Chem
    from rdkit.Chem import AllChem

    print('Preprocessing ZINC250k for novelty calculation')

    df = pd.read_csv('data/zinc250k.csv')
    with open('data/valid_idx_zinc250k.json') as f:
        test_idx = set(json.load(f))
    train_idx = [i for i in range(len(df)) if i not in test_idx]

    train_smiles = df.iloc[train_idx]['smiles']
    train_mols = [Chem.MolFromSmiles(smi) for smi in train_smiles]
    train_smiles = set([Chem.MolToSmiles(mol, isomericSmiles=False) for mol in train_mols])
    train_fps = [AllChem.GetMorganFingerprintAsBitVect((mol), 2, 1024) for mol in train_mols]    
    torch.save((train_smiles, train_fps), 'data/zinc250k_novelty.pt')

COLUMNS = ['SMILES', 'DOCKING', 'QED', 'SA', 'TOTAL']
df = pd.read_csv(args.file_path, names=COLUMNS).iloc[:3000]
n_total_smi = len(df)
print(f'Number of molecules:\t{n_total_smi}')

if args.target == 'parp1':
    hit_thr = 10.
elif args.target == 'fa7':
    hit_thr = 8.5
elif args.target == '5ht1b':
    hit_thr = 8.7845
elif args.target == 'braf':
    hit_thr = 10.3
elif args.target == 'jak2':
    hit_thr = 9.1

df['MOL'] = df['SMILES'].apply(Chem.MolFromSmiles)
df = df.dropna(subset=['MOL'])

get_novelty(df)
print(f'Novelty:\t\t{len(df[df["SIM"] < 0.4]) / n_total_smi}')

df = df.drop_duplicates(subset=['SMILES'])

if 'QED' not in df:
    df['QED'] = reward_qed(df['MOL'])
if 'SA' not in df:
    df['SA'] = reward_sa(df['MOL'])

df = df[df['QED'] > 0.5]
df = df[df['SA'] > 5 / 9]

df2 = df[df['DOCKING'] > hit_thr]
ncircle = get_ncircle(df2)
print(f'#Circle:\t\t{ncircle}')

df = df[df['SIM'] < 0.4]
hit_ratio, top_5_score = get_hit_top5(df)
print(f'Novel hit ratio:\t{hit_ratio}')
print(f'Novel top 5% DS:\t{top_5_score}')
