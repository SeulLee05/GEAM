import random
import copy
import numpy as np
from scipy import sparse
import dgl
import gym
import torch
from utils_sac.docking.docking import DockingVina
from utils_sac.utils import *


def adj2sparse(adj):
    adj = [x*(i+1) for i, x in enumerate(adj)]
    adj = [sparse.dok_matrix(x) for x in adj]
    
    if not all([adj_i is None for adj_i in adj]):
        adj = sparse.dok_matrix(np.sum(adj))
        adj.setdiag(0)   

        all_edges = list(adj.items())
        e_types = np.array([edge[1]-1 for edge in all_edges], dtype=int)
        e = np.transpose(np.array([list(edge[0]) for edge in all_edges]))

        n_edges = len(all_edges)

        e_x = np.zeros((n_edges, 4))
        e_x[np.arange(n_edges),e_types] = 1
        e_x = torch.Tensor(e_x)
        return e, e_x
    else:
        return None


def map_idx(idx, idx_list, mol):
    abs_id = idx_list[idx]
    neigh_idx = mol.GetAtomWithIdx(abs_id).GetNeighbors()[0].GetIdx()
    return neigh_idx 


class MoleculeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def init(self, vocab, target):
        '''
        own init function, since gym does not support passing argument
        '''
        self.starting_smi = 'c1([*:1])c([*:2])ccc([*:3])c1' # benzene
        self.smi = self.starting_smi

        self.mol = Chem.MolFromSmiles(self.smi)
        self.smile_list = []

        self.vocab = vocab
        self.possible_bond_types = np.array([Chem.rdchem.BondType.SINGLE,
                                             Chem.rdchem.BondType.DOUBLE,
                                             Chem.rdchem.BondType.TRIPLE,
                                             Chem.rdchem.BondType.AROMATIC], dtype=object)

        self.d_n = len(vocab['ATOM']) + 18

        self.max_atom = 150
        self.action_space = gym.spaces.MultiDiscrete([20, len(vocab['FRAG']), 20])

        self.counter = 0

        self.predictor = DockingVina(target)

        self.attach_point = Chem.MolFromSmiles('*')
        self.Na = Chem.MolFromSmiles('[Na+]')
        self.K = Chem.MolFromSmiles('[K+]')

    def seed(self, seed):
        np.random.seed(seed=seed)
        random.seed(seed)

    def update_vocab(self, vocab):
        self.vocab = vocab
        self.action_space = gym.spaces.MultiDiscrete([20, len(self.vocab['FRAG']), 20])

    def reset_batch(self):
        self.smile_list = []
    
    def reward_batch(self):
        return reward_vina(self.smile_list, self.predictor)
    
    def reward_batch_vqs(self):
        mols = [Chem.MolFromSmiles(s) for s in self.smile_list]
        rv = reward_vina(self.smile_list, self.predictor)
        rq = reward_qed(mols)
        rs = reward_sa(mols)
        return (rv, rq, rs), np.clip(rv, 0, 20) / 20 * rq * rs

    def step(self, ac):
        info = {}
        self.mol_old = copy.deepcopy(self.mol)
        
        stop = False
        new = False
        
        if get_att_points(self.mol) == [] or self.mol.GetNumAtoms() > 30:
            new = True
        else:
            self._add_motif(ac)

        reward_step = 0.05
        if self.mol.GetNumAtoms() > self.mol_old.GetNumAtoms():
            reward_step += 0.005
        self.counter += 1

        if new:
            reward = 0
            # Only store for obs if attachment point exists in o2
            if get_att_points(self.mol) != []:
                mol_no_att = self.get_final_mol()
                Chem.SanitizeMol(mol_no_att, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                smi_no_att = Chem.MolToSmiles(mol_no_att)
                info['smile'] = smi_no_att
                self.smile_list.append(smi_no_att)
                stop = True
            else:
                stop = False
            self.counter = 0
        else:
            reward = reward_step

        info['stop'] = stop

        ob = self.get_observation()
        return ob, reward, new, info

    def reset(self, smile=None):
        '''
        to avoid error, assume an atom already exists
        :return: ob
        '''
        if smile is not None:
            self.mol = Chem.RWMol(Chem.MolFromSmiles(smile))
            Chem.SanitizeMol(self.mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        else:
            self.smi = self.starting_smi
            self.mol = Chem.MolFromSmiles(self.smi)
        self.counter = 0
        ob = self.get_observation()
        return ob

    def render(self, mode='human', close=False):
        return

    def sample_motif(self):
        cur_mol_atts = get_att_points(self.mol)
        ac1 = np.random.randint(len(cur_mol_atts))
        ac2 = np.random.randint(len(self.vocab['FRAG']))
        ac3 = np.random.randint(len(self.vocab['FRAG_ATT'][ac2]))
        
        a = self.action_space.sample()
        a[0] = ac1
        a[1] = ac2
        a[2] = ac3
        return a

    def _add_motif(self, ac):
        cur_mol = Chem.ReplaceSubstructs(self.mol, self.attach_point, self.Na)[ac[0]]
        motif = self.vocab['FRAG_MOL'][ac[1]]
        att_point = self.vocab['FRAG_ATT'][ac[1]]
        motif_atom = map_idx(ac[2], att_point, motif)
        motif_ = Chem.ReplaceSubstructs(motif, self.attach_point, self.K)[ac[2]]
        motif_ = Chem.DeleteSubstructs(motif_, self.K)
        next_mol = Chem.ReplaceSubstructs(cur_mol, self.Na, motif_, replacementConnectionPoint=motif_atom)[0]
        self.mol = next_mol

    def get_final_smiles_mol(self):
        m = Chem.DeleteSubstructs(self.mol, Chem.MolFromSmiles("*"))
        m = convert_radical_electrons_to_hydrogens(m)
        return m, Chem.MolToSmiles(m, isomericSmiles=True)

    def get_final_mol(self):
        m = Chem.DeleteSubstructs(self.mol, Chem.MolFromSmiles("*"))
        return m
    
    def get_final_mol_ob(self, mol):
        m = Chem.DeleteSubstructs(mol, Chem.MolFromSmiles("*"))
        return m

    def get_observation(self, expert_smi=None):
        """
        ob['adj']:d_e*n*n --- 'E'
        ob['node']:1*n*d_n --- 'F'
        n = atom_num + atom_type_num
        """
        ob = {}

        if expert_smi:
            mol = Chem.MolFromSmiles(expert_smi)
        else:
            ob['att'] = get_att_points(self.mol)
            mol = copy.deepcopy(self.mol)
        
        try:
            Chem.SanitizeMol(mol)
        except:
            pass

        smi = Chem.MolToSmiles(mol)
        F = np.zeros((1, self.max_atom, self.d_n))

        for a in mol.GetAtoms():
            atom_idx = a.GetIdx()
            
            float_array = atom_feature(a, use_atom_meta=True)
            F[0, atom_idx, :] = float_array

        d_e = len(self.possible_bond_types)
        E = np.zeros((d_e, self.max_atom, self.max_atom))

        for b in mol.GetBonds(): 
            begin_idx = b.GetBeginAtomIdx()
            end_idx = b.GetEndAtomIdx()
            bond_type = b.GetBondType()
            float_array = (bond_type == self.possible_bond_types).astype(float)
            try:
                assert float_array.sum() != 0
            except:
                print('error', bond_type)
            E[:, begin_idx, end_idx] = float_array
        
        ob_adj = adj2sparse(E.squeeze())
        ob_node = torch.Tensor(F)
        g = dgl.DGLGraph()

        ob_len = torch.sum(torch.sum(ob_node, dim=-1).bool().float().squeeze(-2), dim=-1)
        g.add_nodes(ob_len)
        if ob_adj is not None and len(ob_adj[0])>0 :
            g.add_edges(ob_adj[0][0], ob_adj[0][1], {'x': ob_adj[1]})
        g.ndata['x'] = ob_node[:, :int(ob_len),:].squeeze(0)
        
        ob['g'] = g
        ob['smi'] = smi
        
        return ob

    def get_observation_mol(self, mol):
        """
        ob['adj']:d_e*n*n --- 'E'
        ob['node']:1*n*d_n --- 'F'
        n = atom_num + atom_type_num
        """
        ob = {}
        ob['att'] = get_att_points(mol)
        
        try:
            Chem.SanitizeMol(mol)
        except:
            pass

        smi = Chem.MolToSmiles(mol)
        F = np.zeros((1, self.max_atom, self.d_n))

        for a in mol.GetAtoms():
            atom_idx = a.GetIdx()
            float_array = atom_feature(a, use_atom_meta=True)
            F[0, atom_idx, :] = float_array

        d_e = len(self.possible_bond_types)
        E = np.zeros((d_e, self.max_atom, self.max_atom))

        for b in mol.GetBonds(): 
            begin_idx = b.GetBeginAtomIdx()
            end_idx = b.GetEndAtomIdx()
            bond_type = b.GetBondType()
            float_array = (bond_type == self.possible_bond_types).astype(float)

            try:
                assert float_array.sum() != 0
            except:
                print('error', bond_type)
            E[:, begin_idx, end_idx] = float_array
        
        ob_adj = adj2sparse(E.squeeze())
        ob_node = torch.Tensor(F)
        g = dgl.DGLGraph()

        ob_len = torch.sum(torch.sum(ob_node, dim=-1).bool().float().squeeze(-2), dim=-1)
        g.add_nodes(ob_len)
        if ob_adj is not None and len(ob_adj[0])>0 :
            g.add_edges(ob_adj[0][0], ob_adj[0][1], {'x': ob_adj[1]})
        g.ndata['x'] = ob_node[:, :int(ob_len),:].squeeze(0)
        
        ob['g'] = g
        ob['smi'] = smi
        return ob
