import random
import more_itertools as mit
import torch
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem


def get_novelty(df):
    if 'FPS' not in df:
        df['FPS'] = [AllChem.GetMorganFingerprintAsBitVect((mol), 2, 1024) for mol in df['MOL']]
    
    train_smiles, train_fps = torch.load('data/zinc250k_novelty.pt')
    
    # hard novelty
    num_novel = len([smi for smi in df['SMILES'] if smi not in train_smiles])

    # soft novelty
    max_sims = []
    for fps in df['FPS']:
        max_sim = max(DataStructs.BulkTanimotoSimilarity(fps, train_fps))
        max_sims.append(max_sim)
    df['SIM'] = max_sims
    
    return num_novel


def get_ncircle(df):
    if 'FPS' not in df:
        df['FPS'] = [AllChem.GetMorganFingerprintAsBitVect((mol), 2, 1024) for mol in df['MOL']]
    return NCircles().measure(df['FPS'])


def similarity_matrix_tanimoto(fps1, fps2):
    similarities = [DataStructs.BulkTanimotoSimilarity(fp, fps2) for fp in fps1]
    return np.array(similarities)


class NCircles():
    def __init__(self, threshold=0.75):
        super().__init__()
        self.sim_mat_func = similarity_matrix_tanimoto
        self.t = threshold
    
    def get_circles(self, args):
        vecs, sim_mat_func, t = args
        
        circs = []
        for vec in vecs:
            if len(circs) > 0:
                dists = 1. - sim_mat_func([vec], circs)
                if dists.min() <= t: continue
            circs.append(vec)
        return circs

    def measure(self, vecs, n_chunk=64):
        for i in range(3):
            vecs_list = [list(c) for c in mit.divide(n_chunk // (2 ** i), vecs)]
            args = zip(vecs_list, 
                       [self.sim_mat_func] * len(vecs_list), 
                       [self.t] * len(vecs_list))
            circs_list = list(map(self.get_circles, args))
            vecs = [c for ls in circs_list for c in ls]
            random.shuffle(vecs)
        vecs = self.get_circles((vecs, self.sim_mat_func, self.t))
        return len(vecs)
    

### code adapted from https://github.com/molecularsets/moses/blob/7b8f83b21a9b7ded493349ec8ef292384ce2bb52/moses/metrics/utils.py#L122
def average_agg_tanimoto(stock_vecs, gen_vecs,
                     batch_size=5000, agg='mean', p=2):
    assert agg in ['max', 'mean'], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j:j + batch_size]).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i:i + batch_size]).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                         y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac ** p
            if agg == 'max':
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto) ** (1 / p)
    return agg_tanimoto.mean()
