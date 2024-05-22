import time
from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
import numpy as np
from rdkit import Chem

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, lr_scheduler

from torch_geometric.data import Batch

from model.ac import GCNActorCritic
from utils_sac.utils import get_att_points, delete_multiple_element
from utils_ga.ga import reproduce
from utils_fgib.utils import get_load_model, get_sanitize_error_frags
from utils_fgib.data import get_graph


class ReplayBuffer:
    def __init__(self, size):
        self.obs_buf = []                                   # o
        self.obs2_buf = []                                  # o2
        self.act_buf = np.zeros((size, 3), dtype=np.int32)  # ac
        self.rew_buf = np.zeros(size, dtype=np.float32)     # r
        self.done_buf = np.zeros(size, dtype=np.float32)    # d
        
        self.ac_prob_buf = []
        self.log_ac_prob_buf = []
        
        self.ac_first_buf = []
        self.ac_second_buf = []
        self.ac_third_buf = []
        self.o_embeds_buf = []
        
        self.ptr, self.size, self.max_size = 0, 0, size
        self.done_location = []

    def store(self, obs, act, rew, next_obs, done, ac_prob, log_ac_prob,
              ac_first_prob, ac_second_hot, ac_third_prob, o_embeds):
        if self.size == self.max_size:
            self.obs_buf.pop(0)
            self.obs2_buf.pop(0)
            
            self.ac_prob_buf.pop(0)
            self.log_ac_prob_buf.pop(0)
            
            self.ac_first_buf.pop(0)
            self.ac_second_buf.pop(0)
            self.ac_third_buf.pop(0)

            self.o_embeds_buf.pop(0)

        self.obs_buf.append(obs)
        self.obs2_buf.append(next_obs)
        
        self.ac_prob_buf.append(ac_prob)
        self.log_ac_prob_buf.append(log_ac_prob)
        
        self.ac_first_buf.append(ac_first_prob)
        self.ac_second_buf.append(ac_second_hot)
        self.ac_third_buf.append(ac_third_prob)
        self.o_embeds_buf.append(o_embeds)

        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        if done:
            self.done_location.append(self.ptr)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def rew_store(self, rew):
        done_location_np = np.array(self.done_location)
        zeros = np.where(rew == 0.0)[0]
        nonzeros = np.where(rew != 0.0)[0]
        zero_ptrs = done_location_np[zeros]

        done_location_np = done_location_np[nonzeros]
        rew = rew[nonzeros]

        if len(self.done_location) > 0:
            self.rew_buf[done_location_np] += rew
            self.done_location = []

        self.act_buf = np.delete(self.act_buf, zero_ptrs, axis=0)
        self.rew_buf = np.delete(self.rew_buf, zero_ptrs)
        self.done_buf = np.delete(self.done_buf, zero_ptrs)
        delete_multiple_element(self.obs_buf, zero_ptrs.tolist())
        delete_multiple_element(self.obs2_buf, zero_ptrs.tolist())

        delete_multiple_element(self.ac_prob_buf, zero_ptrs.tolist())
        delete_multiple_element(self.log_ac_prob_buf, zero_ptrs.tolist())
        
        delete_multiple_element(self.ac_first_buf, zero_ptrs.tolist())
        delete_multiple_element(self.ac_second_buf, zero_ptrs.tolist())
        delete_multiple_element(self.ac_third_buf, zero_ptrs.tolist())

        delete_multiple_element(self.o_embeds_buf, zero_ptrs.tolist())

        self.size = min(self.size - len(zero_ptrs), self.max_size)
        self.ptr = (self.ptr - len(zero_ptrs)) % self.max_size
        
    def sample_batch(self, device, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        obs_batch = [self.obs_buf[idx] for idx in idxs]
        obs2_batch = [self.obs2_buf[idx] for idx in idxs]

        ac_prob_batch = [self.ac_prob_buf[idx] for idx in idxs]
        log_ac_prob_batch = [self.log_ac_prob_buf[idx] for idx in idxs]
        
        ac_first_batch = torch.stack([self.ac_first_buf[idx].to(device) for idx in idxs]).squeeze(1)
        ac_second_batch = torch.stack([self.ac_second_buf[idx].to(device) for idx in idxs]).squeeze(1)
        ac_third_batch = torch.stack([self.ac_third_buf[idx].to(device) for idx in idxs]).squeeze(1)
        o_g_emb_batch = torch.stack([self.o_embeds_buf[idx][2] for idx in idxs]).squeeze(1)

        act_batch = torch.as_tensor(self.act_buf[idxs], dtype=torch.float32).unsqueeze(-1).to(device)
        rew_batch = torch.as_tensor(self.rew_buf[idxs], dtype=torch.float32).to(device)
        done_batch = torch.as_tensor(self.done_buf[idxs], dtype=torch.float32).to(device)

        batch = dict(obs=obs_batch,
                     obs2=obs2_batch,
                     act=act_batch,
                     rew=rew_batch,
                     done=done_batch,
                     ac_prob=ac_prob_batch,
                     log_ac_prob=log_ac_prob_batch,
                     ac_first=ac_first_batch,
                     ac_second=ac_second_batch,
                     ac_third=ac_third_batch,
                     o_g_emb=o_g_emb_batch)
        return batch


def xavier_uniform_init(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)


class SAC:
    def __init__(self, args, vocab, env_fn,
                 replay_size=int(1e6), gamma=0.99, polyak=0.995, train_alpha=True):
        super().__init__()
        
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        self.device = args.device
        self.num_mols = args.num_mols
        self.gamma = gamma
        self.polyak = polyak
        
        tm = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.fname = f'results/{tm}_{args.target}_{args.seed}.csv'
        print(f'\033[92m{self.fname}\033[0m')
        
        self.batch_size = args.batch_size
        self.start_steps = args.start_steps
        self.update_after = args.update_after
        self.update_every = args.update_every
        self.docking_every = int(args.update_every / 2)
        self.train_alpha = train_alpha

        self.env = env_fn
        self.vocab = vocab

        self.obs_dim = args.emb_size * 2
        self.action_dims = [40, len(vocab['FRAG']), 40]
        
        self.target_entropy = args.target_entropy

        self.log_alpha = torch.tensor([np.log(args.init_alpha)], requires_grad=train_alpha) 

        self.ac = GCNActorCritic(self.env, args, vocab).to(args.device)
        self.ac_targ = deepcopy(self.ac).to(args.device).eval()

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        for q in self.ac.parameters():
            q.requires_grad = True

        self.replay_buffer = ReplayBuffer(size=replay_size)

        pi_lr = args.init_pi_lr
        q_lr = args.init_q_lr
        alpha_lr = args.init_alpha_lr
    
        self.pi_params = list(self.ac.pi.parameters()) 
        self.q_params = list(self.ac.q1.parameters()) + list(self.ac.q2.parameters()) + list(self.ac.embed.parameters())
        
        self.pi_optimizer = Adam(self.pi_params, lr=pi_lr, weight_decay=1e-4)
        self.q_optimizer = Adam(self.q_params, lr=q_lr, weight_decay=1e-4)
        self.alpha_optimizer = Adam([self.log_alpha], lr=alpha_lr, eps=1e-4)

        self.q_scheduler = lr_scheduler.ReduceLROnPlateau(self.q_optimizer, factor=0.1, patience=768) 
        self.pi_scheduler = lr_scheduler.ReduceLROnPlateau(self.pi_optimizer, factor=0.1, patience=768)        

        self.alpha_start = self.start_steps
        self.alpha_end = self.start_steps + 30000
        self.alpha_max = args.alpha_max
        self.alpha_min = args.alpha_min
        
        self.population_size = args.population_size
        self.mutation_rate = args.mutation_rate
        self.population = []
        self.population_score = []
        self.ga_smiles_list = []
        
        self.gib = get_load_model(args.gib_path, device=self.device)
        self.max_vocab_update = args.max_vocab_update
        self.max_vocab_size = args.max_vocab_size

        self.t = 0
        self.ac.apply(xavier_uniform_init)

    def update_vocab(self, mol_list, scores_list):
        smiles_list = [Chem.MolToSmiles(m) for m in mol_list]
        batch = []
        for smiles in smiles_list:
            graph = get_graph(smiles)
            if graph is not None:
                batch.append(graph)
        if not batch:
            return
        
        batch = Batch.from_data_list(batch).to(self.device)
        with torch.no_grad():
            p = self.gib(batch, get_w=True)

        i, frag_w_dict, frag_prop_dict = 0, defaultdict(list), defaultdict(list)
        for j, frags in enumerate(batch.frags):
            for frag in frags:
                frag_w_dict[frag].append(p[i].item())
                frag_prop_dict[frag].append(scores_list[j])
                i += 1
        
        error_frags = get_sanitize_error_frags(frag_w_dict)
        for frag in error_frags:
            del frag_w_dict[frag]
            del frag_prop_dict[frag]

        frag_num_dict = {}
        for frag in frag_w_dict:
            frag_num_dict[frag] = Chem.MolFromSmiles(frag).GetNumAtoms()
        
        scores = [(np.array(frag_prop_dict[k]) * np.array(frag_w_dict[k])).mean() / np.sqrt(frag_num_dict[k]) for k in frag_w_dict]
        frag_tuples = list(zip(frag_w_dict, scores))
        frag_tuples = sorted(frag_tuples, key=lambda x: x[1], reverse=True)[:self.max_vocab_update]
        frag_tuples = [(frag, score) for frag, score in frag_tuples
                       if frag not in self.vocab['FRAG']]
        
        self.vocab['FRAG_QUEUE'].extend(frag_tuples)
        self.vocab['FRAG_QUEUE'] = sorted(self.vocab['FRAG_QUEUE'],
                                          key=lambda x: x[1], reverse=True)[:self.max_vocab_size]
        self.vocab['FRAG'] = [frag for frag, score in self.vocab['FRAG_QUEUE']]
        self.vocab['FRAG_MOL'] = [Chem.MolFromSmiles(frag) for frag in self.vocab['FRAG']]
        self.vocab['FRAG_ATT'] = [get_att_points(mol) for mol in self.vocab['FRAG_MOL']]
        
        self.action_dims = [40, len(self.vocab['FRAG']), 40]
        self.env.update_vocab(self.vocab)
        self.ac.pi.update_vocab(self.vocab)
        torch.cuda.empty_cache()
    
    def compute_loss_q(self, data):
        ac_first, ac_second, ac_third = data['ac_first'], data['ac_second'], data['ac_third']             
        self.ac.q1.train()
        self.ac.q2.train()
        o = data['obs']
        _, _, o_g_emb = self.ac.embed(o)
        q1 = self.ac.q1(o_g_emb, ac_first, ac_second, ac_third).squeeze()
        q2 = self.ac.q2(o_g_emb.detach(), ac_first, ac_second, ac_third).squeeze()

        # Target actions come from *current* policy
        o2 = data['obs2']
        r, d = data['rew'], data['done']
        
        with torch.no_grad():
            o2_g, o2_n_emb, o2_g_emb = self.ac.embed(o2)
            cands = self.ac.embed(self.ac.pi.cand)
            a2, (a2_prob, log_a2_prob), (ac2_first, ac2_second, ac2_third) = self.ac.pi(o2_g_emb, o2_n_emb, o2_g, cands)
            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2_g_emb, ac2_first, ac2_second, ac2_third)
            q2_pi_targ = self.ac_targ.q2(o2_g_emb, ac2_first, ac2_second, ac2_third)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ).squeeze() 
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        return loss_q

    def compute_loss_pi(self, data):
        with torch.no_grad():
            o_embeds = self.ac.embed(data['obs'])   
            o_g, o_n_emb, o_g_emb = o_embeds
            cands = self.ac.embed(self.ac.pi.cand)

        _, (ac_prob, log_ac_prob), (ac_first, ac_second, ac_third) = \
            self.ac.pi(o_g_emb, o_n_emb, o_g, cands)

        q1_pi = self.ac.q1(o_g_emb, ac_first, ac_second, ac_third)
        q2_pi = self.ac.q2(o_g_emb, ac_first, ac_second, ac_third)
        q_pi = torch.min(q1_pi, q2_pi)

        ac_prob_sp = torch.split(ac_prob, self.action_dims, dim=1)
        log_ac_prob_sp = torch.split(log_ac_prob, self.action_dims, dim=1)
        
        loss_policy = torch.mean(-q_pi)        

        # Entropy-regularized policy loss
        alpha = min(self.log_alpha.exp().item(), self.alpha_max)
        alpha = max(self.log_alpha.exp().item(), self.alpha_min)

        loss_entropy = 0
        loss_alpha = 0
        
        ac_prob_comb = torch.einsum('by, bz->byz', ac_prob_sp[1], ac_prob_sp[2]).reshape(self.batch_size, -1) # (bs , 73 x 40)
        ac_prob_comb = torch.einsum('bx, bz->bxz', ac_prob_sp[0], ac_prob_comb).reshape(self.batch_size, -1) # (bs , 40 x 73 x 40)
        # order by (a1, b1, c1) (a1, b1, c2)! Be advised!
        
        log_ac_prob_comb = log_ac_prob_sp[0].reshape(self.batch_size, self.action_dims[0], 1, 1).repeat(
                                    1, 1, self.action_dims[1], self.action_dims[2]).reshape(self.batch_size, -1)\
                            + log_ac_prob_sp[1].reshape(self.batch_size, 1, self.action_dims[1], 1).repeat(
                                    1, self.action_dims[0], 1, self.action_dims[2]).reshape(self.batch_size, -1)\
                            + log_ac_prob_sp[2].reshape(self.batch_size, 1, 1, self.action_dims[2]).repeat(
                                    1, self.action_dims[0], self.action_dims[1], 1).reshape(self.batch_size, -1)
        loss_entropy = (alpha * ac_prob_comb * log_ac_prob_comb).sum(dim=1).mean()
        loss_alpha = -(self.log_alpha.to(self.device) * \
                        ((ac_prob_comb * log_ac_prob_comb).sum(dim=1) + self.target_entropy).detach()).mean()

        return loss_entropy, loss_policy, loss_alpha

    def update(self, data):
        # First run one gradient descent step for Q1 and Q2
        ave_pi_grads, ave_q_grads = [], []
        
        loss_q = self.compute_loss_q(data)
        self.q_optimizer.zero_grad()
        loss_q.backward()
        clip_grad_norm_(self.q_params, 5)
        for q in list(self.q_params):
            ave_q_grads.append(q.grad.abs().mean().item())
        
        self.q_optimizer.step()
        self.q_scheduler.step(loss_q)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for q in self.q_params:
            q.requires_grad = False

        loss_entropy, loss_policy, loss_alpha = self.compute_loss_pi(data)
        loss_pi = loss_entropy + loss_policy
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        clip_grad_norm_(self.pi_params, 5)
        for p in self.pi_params:
            ave_pi_grads.append(p.grad.abs().mean().item())
        
        self.pi_optimizer.step()
        self.pi_scheduler.step(loss_policy)
        
        if self.train_alpha:
            if self.alpha_start <= self.t < self.alpha_end:
                self.alpha_optimizer.zero_grad()
                loss_alpha.backward()
                self.alpha_optimizer.step()
        
        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True
        
        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            self.ac_targ.load_state_dict(self.ac.state_dict())
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def run(self):
        num_generated = 0
        pbar = tqdm(total=self.num_mols)
        o = self.env.reset()

        while True:
            with torch.no_grad():
                cands = self.ac.embed(self.ac.pi.cand)
                o_embeds = self.ac.embed([o])
                o_g, o_n_emb, o_g_emb = o_embeds

                if self.t >= self.start_steps:
                    ac, (ac_prob, log_ac_prob), (ac_first, ac_second, ac_third) = \
                    self.ac.pi(o_g_emb, o_n_emb, o_g, cands)
                else:
                    ac = self.env.sample_motif()[np.newaxis]
                    (ac_prob, log_ac_prob), (ac_first, ac_second, ac_third) = \
                    self.ac.pi.sample(ac[0], o_g_emb, o_n_emb, o_g, cands)

            o2, r, d, info = self.env.step(ac[0])

            r_d = info['stop']
            # Only store observations where attachment point exists in o2
            if any(o2['att']):
                if type(ac) == np.ndarray:
                    self.replay_buffer.store(o, ac, r, o2, r_d,
                                             ac_prob, log_ac_prob, ac_first, ac_second, ac_third,
                                             o_embeds)
                else:
                    self.replay_buffer.store(o, ac.detach().cpu().numpy(), r, o2, r_d,
                                             ac_prob, log_ac_prob, ac_first, ac_second, ac_third,
                                             o_embeds)

            # Super critical, easy to overlook step: make sure to update most recent observation!
            o = o2

            # End of trajectory handling
            if get_att_points(self.env.mol) == []:  # Temporally force attachment calculation
                d = True
            if not any(o2['att']):
                d = True

            if d:
                o = self.env.reset()

                # GA reproduce
                if self.t >= self.start_steps and len(self.population) >= 2:
                    offspring = reproduce(self.population, self.population_score, self.mutation_rate)
                    if offspring is not None:
                        self.ga_smiles_list.append(Chem.MolToSmiles(offspring))

            if self.t > 1 and self.t % self.docking_every == 0 and self.env.smile_list != []:
                n_sac_smi = len(self.env.smile_list)
                self.env.smile_list += self.ga_smiles_list
                n_smi = len(self.env.smile_list)
                if n_smi > 0:
                    rews, ext_rew = self.env.reward_batch_vqs()

                    r_batch = ext_rew[:n_sac_smi]
                    self.replay_buffer.rew_store(r_batch)

                    with open(self.fname, 'a') as f:
                        for i in range(n_smi):
                            str = f'{self.env.smile_list[i]},'
                            for rew in rews: str += f'{rew[i]},'
                            str += (f'{ext_rew[i]}' + '\n')
                            f.write(str)
                    
                    mols = [Chem.MolFromSmiles(s) for s in self.env.smile_list]

                    # update vocab
                    if self.t >= self.start_steps:
                        self.update_vocab(mols[n_sac_smi:], ext_rew[n_sac_smi:])

                    # GA population handling
                    self.population.extend(mols)
                    self.population_score.extend(ext_rew)
                    population_tuples = list(zip(self.population, self.population_score))
                    population_tuples = sorted(population_tuples, key=lambda x: x[1], reverse=True)[:self.population_size]
                    self.population = [t[0] for t in population_tuples]
                    self.population_score = [t[1] for t in population_tuples]

                    num_generated += n_smi
                    pbar.update(n_smi)
                    
                    if num_generated >= self.num_mols:
                        pbar.close()
                        break
                    
                    self.env.reset_batch()
                    self.ga_smiles_list = []

            # Update handling
            if self.t >= self.update_after and self.t % self.update_every == 0:
                for j in range(self.update_every):
                    batch = self.replay_buffer.sample_batch(self.device, self.batch_size)
                    self.update(data=batch)
            
            self.t += 1
