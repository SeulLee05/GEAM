import os
import torch
import gym
import argparse

from model.sac import SAC
from utils_sac.utils import set_seed, get_vocab

import warnings
warnings.filterwarnings(action='ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu_id', type=int, default=-1)
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-t', '--target', type=str, default='parp1',
                        choices=['parp1', 'fa7', '5ht1b', 'braf', 'jak2'])
    parser.add_argument('-v', '--vocab_path', type=str, required=True)
    
    parser.add_argument('--num_mols', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--start_steps', type=int, default=4000)
    parser.add_argument('--update_after', type=int, default=3000)
    parser.add_argument('--update_every', type=int, default=256)
    
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--num_layer', type=int, default=3)
    
    parser.add_argument('--tau', type=float, default=1e-1)
    parser.add_argument('--target_entropy', type=float, default=1.)
    parser.add_argument('--init_alpha', type=float, default=1.)
    parser.add_argument('--init_pi_lr', type=float, default=1e-4)
    parser.add_argument('--init_q_lr', type=float, default=1e-4)
    parser.add_argument('--init_alpha_lr', type=float, default=5e-4)
    parser.add_argument('--alpha_max', type=float, default=20.)
    parser.add_argument('--alpha_min', type=float, default=.05)

    parser.add_argument('--population_size', type=int, default=100)
    parser.add_argument('--mutation_rate', type=float, default=0.1)

    parser.add_argument('-m', '--gib_path', type=str, required=True)
    parser.add_argument('--max_vocab_update', type=int, default=50)
    parser.add_argument('--max_vocab_size', type=int, default=1000)
    args = parser.parse_args()
    print(args)
    
    if args.gpu_id >= 0:
        args.device = torch.device(f'cuda:{args.gpu_id}')
    else:
        args.device = torch.device('cpu')
        torch.set_num_threads(256)
    
    if not os.path.exists('results'):
        os.makedirs('results')

    gym.envs.registration.register(id='molecule-v0', entry_point='utils_sac.env:MoleculeEnv')
    set_seed(args.seed)

    vocab = get_vocab(args.vocab_path)

    env = gym.make('molecule-v0')
    env.init(vocab=vocab, target=args.target)
    env.seed(args.seed)

    sac = SAC(args, vocab, env)
    sac.run()
    env.close()


if __name__ == '__main__':
    main()
