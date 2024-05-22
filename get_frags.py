import argparse
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import torch
from rdkit import RDLogger, Chem
from utils_fgib.utils import get_load_model, get_loader, get_sanitize_error_frags
RDLogger.DisableLog('rdApp.*')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu_id", type=int, default=-1)
    parser.add_argument("-t", "--target", type=str, default="parp1",
                        choices=['parp1', 'fa7', '5ht1b', 'braf', 'jak2',
                                 'amlodipine_mpo', 'fexofenadine_mpo',
                                 'osimertinib_mpo', 'perindopril_mpo',
                                 'ranolazine_mpo', 'sitagliptin_mpo', 'zaleplon_mpo'])
    parser.add_argument("-m", "--gib_path", type=str, required=True)
    parser.add_argument("-v", "--vocab_path", type=str, required=True)
    parser.add_argument("-s", "--vocab_size", type=int, default=300)
    args = parser.parse_args()

    if args.gpu_id >= 0:
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')

    model = get_load_model(args.gib_path, device=device)

    train_loader, _ = get_loader(target=args.target, batch_size=1)

    frag_w_dict, frag_prop_dict = defaultdict(list), defaultdict(list)
    for graph, value in tqdm(train_loader):
        with torch.no_grad():
            w = model(graph.to(device), get_w=True)

        for frag, frag_w in zip(graph[0].frags, w):
            frag_w_dict[frag].append(frag_w.item())
            frag_prop_dict[frag].append(value.item())

    error_frags = get_sanitize_error_frags(frag_w_dict)
    for frag in error_frags:
        del frag_w_dict[frag]
        del frag_prop_dict[frag]
    
    frag_num_dict = {}
    for frag in frag_w_dict:
        frag_num_dict[frag] = Chem.MolFromSmiles(frag).GetNumAtoms()

    scores = [(np.array(frag_prop_dict[k]) * np.array(frag_w_dict[k])).mean() / np.sqrt(frag_num_dict[k]) for k in frag_w_dict]
    frag_tuples = list(zip(frag_w_dict, scores))
    frag_tuples = sorted(frag_tuples, key=lambda x: x[1], reverse=True)[:args.vocab_size]
    with open(args.vocab_path, 'w') as f:
        f.writelines([f'{frag},{score}\n' for frag, score in frag_tuples])
