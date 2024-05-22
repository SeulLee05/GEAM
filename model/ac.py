from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn.pytorch.glob import SumPooling

from rdkit import Chem
from utils_sac.utils import ecfp


msg = fn.copy_src(src='x', out='m')


def reduce_mean(nodes):
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'x': accum}


def reduce_sum(nodes):
    accum = torch.sum(nodes.mailbox['m'], 1)
    return {'x': accum}  


class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, agg="sum", is_normalize=False, residual=True):
        super().__init__()
        self.residual = residual
        assert agg in ["sum", "mean"], "Wrong agg type"
        self.agg = agg
        self.is_normalize = is_normalize
        self.linear1 = nn.Linear(in_channels, out_channels, bias=False)
        self.activation = nn.ReLU()

    def forward(self, g):
        h_in = g.ndata['x']
        if self.agg == "sum":
            g.update_all(msg, reduce_sum)
        elif self.agg == "mean":
            g.update_all(msg, reduce_mean)
        h = self.linear1(g.ndata['x'])
        h = self.activation(h)
        if self.is_normalize:
            h = F.normalize(h, p=2, dim=1)
        if self.residual:
            h += h_in
        return h


class GCNPredictor(nn.Module):
    def __init__(self, args, atom_vocab):
        super().__init__()
        self.embed = GCNEmbed(args, atom_vocab)
        self.pred_layer = nn.Sequential(
                    nn.Linear(args.emb_size*2, args.emb_size, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(args.emb_size, 1, bias=True))

    def forward(self, o):
        _, _, graph_emb = self.embed(o)
        pred = self.pred_layer(graph_emb)
        return pred


class GCNQFunction(nn.Module):
    def __init__(self, args, override_seed=False):
        super().__init__()
        if override_seed:
            seed = args.seed + 1
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.batch_size = args.batch_size
        self.device = args.device
        self.emb_size = args.emb_size
        self.d = 3 * args.emb_size + 80

        self.ecfp_emb = nn.Linear(1024, args.emb_size)
        self.qpred_layer = nn.Sequential(
                            nn.Linear(self.d, int(self.d // 2), bias=False),
                            nn.ReLU(inplace=False),
                            nn.Linear(int(self.d // 2), 1, bias=True))
    
    def forward(self, graph_emb, ac_first_prob, ac_second_desc, ac_third_prob):
        ecfp_emb = self.ecfp_emb(ac_second_desc)
        emb_state_action = torch.cat([graph_emb, ac_first_prob, ecfp_emb, ac_third_prob], dim=-1).contiguous()
        qpred = self.qpred_layer(emb_state_action)
        return qpred


class SFSPolicy(nn.Module):
    def __init__(self, env, args, frag_vocab_mol):
        super().__init__()
        self.device = args.device
        self.batch_size = args.batch_size
        self.frag_vocab_mol = frag_vocab_mol
        self.emb_size = args.emb_size
        self.tau = args.tau
        
        # init candidate atoms
        self.bond_type_num = 4

        self.env = env  # env utilized to init cand motif mols
        
        self.cand = self.create_candidate_motifs()
        self.motif_type_num = len(self.cand)
        self.cand_desc = torch.Tensor([ecfp(Chem.MolFromSmiles(x['smi'])) 
                                for x in self.cand]).to(self.device)
        self.ac3_att_len = torch.LongTensor([len(x['att']) 
                                for x in self.cand]).to(self.device)
        self.ac3_att_mask = torch.cat([torch.LongTensor([i]*len(x['att'])) 
                                for i, x in enumerate(self.cand)], dim=0).to(self.device)

        self.action1_layers = nn.ModuleList([nn.Bilinear(2*args.emb_size, 2*args.emb_size, args.emb_size).to(self.device),
                                nn.Linear(2*args.emb_size, args.emb_size, bias=False).to(self.device),
                                nn.Linear(2*args.emb_size, args.emb_size, bias=False).to(self.device), 
                                nn.Sequential(
                                nn.Linear(args.emb_size, args.emb_size//2, bias=False),
                                nn.ReLU(inplace=False),
                                nn.Linear(args.emb_size//2, 1, bias=True)).to(self.device)])
                       
        self.action2_layers = nn.ModuleList([nn.Bilinear(1024, args.emb_size, args.emb_size).to(self.device),
                                nn.Linear(1024, args.emb_size, bias=False).to(self.device),
                                nn.Linear(args.emb_size, args.emb_size, bias=False).to(self.device), 
                                nn.Sequential(
                                nn.Linear(args.emb_size, args.emb_size, bias=False),
                                nn.ReLU(inplace=False),
                                nn.Linear(args.emb_size, args.emb_size, bias=True),
                                nn.ReLU(inplace=False),
                                nn.Linear(args.emb_size, 1, bias=True))])

        self.action3_layers = nn.ModuleList([nn.Bilinear(2*args.emb_size, 2*args.emb_size, args.emb_size).to(self.device),
                                nn.Linear(2*args.emb_size, args.emb_size, bias=False).to(self.device),
                                nn.Linear(2*args.emb_size, args.emb_size, bias=False).to(self.device),
                                nn.Sequential(
                                nn.Linear(args.emb_size, args.emb_size//2, bias=False),
                                nn.ReLU(inplace=False),
                                nn.Linear(args.emb_size//2, 1, bias=True)).to(self.device)])

        # Zero padding with max number of actions
        self.max_action = 40    # max atoms
        
    def update_vocab(self, vocab):
        self.frag_vocab_mol = vocab['FRAG_MOL']
        self.cand = self.create_candidate_motifs()
        self.motif_type_num = len(self.cand)
        self.cand_desc = torch.Tensor([ecfp(Chem.MolFromSmiles(x['smi'])) 
                                for x in self.cand]).to(self.device)
        self.ac3_att_len = torch.LongTensor([len(x['att']) 
                                for x in self.cand]).to(self.device)
        self.ac3_att_mask = torch.cat([torch.LongTensor([i]*len(x['att'])) 
                                for i, x in enumerate(self.cand)], dim=0).to(self.device)

    def create_candidate_motifs(self):
        motif_gs = [self.env.get_observation_mol(mol) for mol in self.frag_vocab_mol]
        return motif_gs

    def gumbel_softmax(self, logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1, \
                    g_ratio: float = 1e-3) -> torch.Tensor:
        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        gumbels = (logits + gumbels * g_ratio) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)
        
        if hard:
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        return ret

    def forward(self, graph_emb, node_emb, g, cands):
        """
        graph_emb : bs x hidden_dim
        node_emb : (bs x num_nodes) x hidden_dim)
        g: batched graph
        att: indexs of attachment points, list of list
        """
        g.ndata['node_emb'] = node_emb
        cand_g, cand_node_emb, cand_graph_emb = cands

        # Only acquire node embeddings with attatchment points
        ob_len = g.batch_num_nodes().tolist()
        att_mask = g.ndata['att_mask'] # used to select att embs from node embs
        
        if g.batch_size != 1:
            att_mask_split = torch.split(att_mask, ob_len, dim=0)
            att_len = [torch.sum(x, dim=0) for x in att_mask_split]
        else:
            att_len = torch.sum(att_mask, dim=-1) # used to torch.split for att embs

        cand_att_mask = cand_g.ndata['att_mask']

        # =============================== 
        # step 1 : where to add
        # =============================== 
        # select only nodes with attachment points
        att_emb = torch.masked_select(node_emb , att_mask.unsqueeze(-1))
        att_emb = att_emb.view(-1, 2*self.emb_size)
        
        if g.batch_size != 1:
            graph_expand = torch.cat([graph_emb[i].unsqueeze(0).repeat(att_len[i],1) for i in range(g.batch_size)], dim=0).contiguous()
        else:
            graph_expand = graph_emb.repeat(att_len, 1)

        att_emb = self.action1_layers[0](att_emb, graph_expand) + self.action1_layers[1](att_emb) \
                    + self.action1_layers[2](graph_expand)
        logits_first = self.action1_layers[3](att_emb)

        if g.batch_size != 1:
            ac_first_prob = [torch.softmax(logit, dim=0)
                            for i, logit in enumerate(torch.split(logits_first, att_len, dim=0))]
            ac_first_prob = [p+1e-8 for p in ac_first_prob]
            log_ac_first_prob = [x.log() for x in ac_first_prob]
        else:
            ac_first_prob = torch.softmax(logits_first, dim=0) + 1e-8
            log_ac_first_prob = ac_first_prob.log()

        if g.batch_size != 1:
            first_stack = []
            first_ac_stack = []
            for i, node_emb_i in enumerate(torch.split(att_emb, att_len, dim=0)):
                ac_first_hot_i = self.gumbel_softmax(ac_first_prob[i], tau=self.tau, hard=True, dim=0).transpose(0,1)
                ac_first_i = torch.argmax(ac_first_hot_i, dim=-1)
                first_stack.append(torch.matmul(ac_first_hot_i, node_emb_i))
                first_ac_stack.append(ac_first_i)

            emb_first = torch.stack(first_stack, dim=0).squeeze(1)
            ac_first = torch.stack(first_ac_stack, dim=0).squeeze(1)
            
            ac_first_prob = torch.cat([
                                torch.cat([ac_first_prob_i, ac_first_prob_i.new_zeros(
                                    max(self.max_action - ac_first_prob_i.size(0), 0), 1)]
                                        , 0).contiguous().view(1,self.max_action)
                                for i, ac_first_prob_i in enumerate(ac_first_prob)], dim=0).contiguous()

            log_ac_first_prob = torch.cat([
                                    torch.cat([log_ac_first_prob_i, log_ac_first_prob_i.new_zeros(
                                        max(self.max_action - log_ac_first_prob_i.size(0), 0), 1)]
                                            , 0).contiguous().view(1,self.max_action)
                                    for i, log_ac_first_prob_i in enumerate(log_ac_first_prob)], dim=0).contiguous()
            
        else:
            ac_first_hot = self.gumbel_softmax(ac_first_prob, tau=self.tau, hard=True, dim=0).transpose(0,1)
            ac_first = torch.argmax(ac_first_hot, dim=-1)
            emb_first = torch.matmul(ac_first_hot, att_emb)
            ac_first_prob = torch.cat([ac_first_prob, ac_first_prob.new_zeros(
                            max(self.max_action - ac_first_prob.size(0), 0), 1)]
                                , 0).contiguous().view(1,self.max_action)
            log_ac_first_prob = torch.cat([log_ac_first_prob, log_ac_first_prob.new_zeros(
                            max(self.max_action - log_ac_first_prob.size(0), 0), 1)]
                                , 0).contiguous().view(1,self.max_action)

        # ===============================
        # step 2 : which motif to add - Using Descriptors
        # ===============================
        emb_first_expand = emb_first.view(-1, 1, self.emb_size).repeat(1, self.motif_type_num, 1)
        cand_expand = self.cand_desc.unsqueeze(0).repeat(g.batch_size, 1, 1)
        
        emb_cat = self.action2_layers[0](cand_expand, emb_first_expand) + \
                    self.action2_layers[1](cand_expand) + self.action2_layers[2](emb_first_expand)

        logit_second = self.action2_layers[3](emb_cat).squeeze(-1)
        ac_second_prob = F.softmax(logit_second, dim=-1) + 1e-8
        log_ac_second_prob = ac_second_prob.log()
        
        ac_second_hot = self.gumbel_softmax(ac_second_prob, tau=self.tau, hard=True, g_ratio=1e-3)                                    
        emb_second = torch.matmul(ac_second_hot, cand_graph_emb)
        ac_second = torch.argmax(ac_second_hot, dim=-1)
        ac_second_desc = torch.matmul(ac_second_hot, self.cand_desc)

        # ===============================
        # step 4 : where to add on motif
        # ===============================
        # Select att points from candidate
        cand_att_emb = torch.masked_select(cand_node_emb, cand_att_mask.unsqueeze(-1))
        cand_att_emb = cand_att_emb.view(-1, 2*self.emb_size)

        ac3_att_mask = self.ac3_att_mask.repeat(g.batch_size, 1) # bs x (num cands * num att size)
        ac3_att_mask = torch.where(ac3_att_mask==ac_second.view(-1, 1),
                            1, 0).view(g.batch_size, -1) # (num_cands * num_nodes)
        ac3_att_mask = ac3_att_mask.bool()

        ac3_cand_emb = torch.masked_select(cand_att_emb.view(1, -1, 2*self.emb_size), 
                                ac3_att_mask.view(g.batch_size, -1, 1)).view(-1, 2*self.emb_size)#.view(1, -1, 2*self.emb_size)
        
        ac3_att_len = torch.index_select(self.ac3_att_len, 0, ac_second).tolist()
        emb_second_expand = torch.cat([emb_second[i].unsqueeze(0).repeat(ac3_att_len[i],1) for i in range(g.batch_size)]).contiguous()

        emb_cat_ac3 = self.action3_layers[0](emb_second_expand, ac3_cand_emb) + self.action3_layers[1](emb_second_expand) \
                  + self.action3_layers[2](ac3_cand_emb)
        
        logits_third = self.action3_layers[3](emb_cat_ac3)

        # predict logit
        if g.batch_size != 1:
            ac_third_prob = [torch.softmax(logit, dim=-1)
                            for i, logit in enumerate(torch.split(logits_third.squeeze(-1), ac3_att_len, dim=0))]
            ac_third_prob = [p+1e-8 for p in ac_third_prob]
            log_ac_third_prob = [x.log() for x in ac_third_prob]
        else:
            logits_third = logits_third.transpose(1,0)
            ac_third_prob = torch.softmax(logits_third, dim=-1) + 1e-8
            log_ac_third_prob = ac_third_prob.log()
        
        # gumbel softmax sampling and zero-padding
        if g.batch_size != 1:
            third_stack = []
            third_ac_stack = []
            for i, node_emb_i in enumerate(torch.split(emb_cat_ac3, ac3_att_len, dim=0)):
                ac_third_hot_i = self.gumbel_softmax(ac_third_prob[i], tau=self.tau, hard=True, dim=-1)
                ac_third_i = torch.argmax(ac_third_hot_i, dim=-1)
                third_stack.append(torch.matmul(ac_third_hot_i, node_emb_i))
                third_ac_stack.append(ac_third_i)

                del ac_third_hot_i
            ac_third = torch.stack(third_ac_stack, dim=0)
            ac_third_prob = torch.cat([
                                torch.cat([ac_third_prob_i, ac_third_prob_i.new_zeros(
                                    self.max_action - ac_third_prob_i.size(0))]
                                        , dim=0).contiguous().view(1,self.max_action)
                                for i, ac_third_prob_i in enumerate(ac_third_prob)], dim=0).contiguous()
            
            log_ac_third_prob = torch.cat([
                                    torch.cat([log_ac_third_prob_i, log_ac_third_prob_i.new_zeros(
                                        self.max_action - log_ac_third_prob_i.size(0))]
                                            , 0).contiguous().view(1,self.max_action)
                                    for i, log_ac_third_prob_i in enumerate(log_ac_third_prob)], dim=0).contiguous()

        else:
            ac_third_hot = self.gumbel_softmax(ac_third_prob, tau=self.tau, hard=True, dim=-1)
            ac_third = torch.argmax(ac_third_hot, dim=-1)
            
            ac_third_prob = torch.cat([ac_third_prob, ac_third_prob.new_zeros(
                                        1, self.max_action - ac_third_prob.size(1))] 
                                , -1).contiguous()
            log_ac_third_prob = torch.cat([log_ac_third_prob, log_ac_third_prob.new_zeros(
                                        1, self.max_action - log_ac_third_prob.size(1))]
                                , -1).contiguous()

        ac_prob = torch.cat([ac_first_prob, ac_second_prob, ac_third_prob], dim=1).contiguous()
        log_ac_prob = torch.cat([log_ac_first_prob,
                            log_ac_second_prob, log_ac_third_prob], dim=1).contiguous()
        ac = torch.stack([ac_first, ac_second, ac_third], dim=1)

        return ac, (ac_prob, log_ac_prob), (ac_first_prob, ac_second_desc, ac_third_prob)
    
    def sample(self, ac, graph_emb, node_emb, g, cands):
        g.ndata['node_emb'] = node_emb
        cand_g, cand_node_emb, cand_graph_emb = cands 

        # Only acquire node embeddings with attachment points
        att_mask = g.ndata['att_mask']          # used to select att embs from node embs
        att_len = torch.sum(att_mask, dim=-1)   # used to torch.split for att embs

        cand_att_mask = cand_g.ndata['att_mask']

        # =============================== 
        # step 1 : where to add
        # =============================== 
        # select only nodes with attachment points
        att_emb = torch.masked_select(node_emb, att_mask.unsqueeze(-1))
        att_emb = att_emb.view(-1, 2*self.emb_size)
        graph_expand = graph_emb.repeat(att_len, 1)
        
        att_emb = self.action1_layers[0](att_emb, graph_expand) + self.action1_layers[1](att_emb) \
                    + self.action1_layers[2](graph_expand)
        logits_first = self.action1_layers[3](att_emb).transpose(1,0)
            
        ac_first_prob = torch.softmax(logits_first, dim=-1) + 1e-8
        
        log_ac_first_prob = ac_first_prob.log()
        ac_first_prob = torch.cat([ac_first_prob, ac_first_prob.new_zeros(1,
                        max(self.max_action - ac_first_prob.size(1),0))]
                            , 1).contiguous()
        
        log_ac_first_prob = torch.cat([log_ac_first_prob, log_ac_first_prob.new_zeros(1,
                        max(self.max_action - log_ac_first_prob.size(1),0))]
                            , 1).contiguous()
        emb_first = att_emb[ac[0]].unsqueeze(0)
        
        # ===============================
        # step 2 : which motif to add
        # ===============================
        emb_first_expand = emb_first.repeat(1, self.motif_type_num, 1)
        cand_expand = self.cand_desc.unsqueeze(0).repeat(g.batch_size, 1, 1)     
        
        emb_cat = self.action2_layers[0](cand_expand, emb_first_expand) + \
                    self.action2_layers[1](cand_expand) + self.action2_layers[2](emb_first_expand)
        
        logit_second = self.action2_layers[3](emb_cat).squeeze(-1)
        ac_second_prob = F.softmax(logit_second, dim=-1) + 1e-8
        log_ac_second_prob = ac_second_prob.log()
        
        ac_second_hot = self.gumbel_softmax(ac_second_prob, tau=self.tau, hard=True, g_ratio=1e-3)                                    
        emb_second = torch.matmul(ac_second_hot, cand_graph_emb)
        ac_second_desc = torch.matmul(ac_second_hot, self.cand_desc)

        # ===============================  
        # step 3 : where to add on motif
        # ===============================
        # Select att points from candidates
        cand_att_emb = torch.masked_select(cand_node_emb, cand_att_mask.unsqueeze(-1))
        cand_att_emb = cand_att_emb.view(-1, 2*self.emb_size)

        ac3_att_mask = self.ac3_att_mask.repeat(g.batch_size, 1) # bs x (num cands * num att size)
        # torch where currently does not support cpu ops  
        
        ac3_att_mask = torch.where(ac3_att_mask==ac[1], 
                            1, 0).view(g.batch_size, -1) # (num_cands * num_nodes)
        ac3_att_mask = ac3_att_mask.bool()

        ac3_cand_emb = torch.masked_select(cand_att_emb.view(1, -1, 2*self.emb_size), 
                                ac3_att_mask.view(g.batch_size, -1, 1)).view(-1, 2*self.emb_size)
        
        ac3_att_len = self.ac3_att_len[ac[1]]
        emb_second_expand = emb_second.repeat(ac3_att_len,1)
        emb_cat_ac3 = self.action3_layers[0](emb_second_expand, ac3_cand_emb) + self.action3_layers[1](emb_second_expand) \
                  + self.action3_layers[2](ac3_cand_emb)

        logits_third = self.action3_layers[3](emb_cat_ac3)
        logits_third = logits_third.transpose(1,0)
        ac_third_prob = torch.softmax(logits_third, dim=-1) + 1e-8
        log_ac_third_prob = ac_third_prob.log()

        # gumbel softmax sampling and zero-padding
        ac_third_prob = torch.cat([ac_third_prob, ac_third_prob.new_zeros(
                                        1, self.max_action - ac_third_prob.size(1))] 
                                , -1).contiguous()
        log_ac_third_prob = torch.cat([log_ac_third_prob, log_ac_third_prob.new_zeros(
                                        1, self.max_action - log_ac_third_prob.size(1))]
                                , -1).contiguous()

        # ==== concat everything ====
        ac_prob = torch.cat([ac_first_prob, ac_second_prob, ac_third_prob], dim=1).contiguous()
        log_ac_prob = torch.cat([log_ac_first_prob, 
                            log_ac_second_prob, log_ac_third_prob], dim=1).contiguous()

        return (ac_prob, log_ac_prob), (ac_first_prob, ac_second_desc, ac_third_prob)
        

class GCNEmbed(nn.Module):
    def __init__(self, args, atom_vocab):
        super().__init__()

        self.device = args.device
        self.bond_type_num = 4
        self.d_n = len(atom_vocab) + 18
        
        self.emb_size = args.emb_size * 2
        in_channels = 8
        self.emb_linear = nn.Linear(self.d_n, in_channels, bias=False)

        self.gcn_layers = nn.ModuleList([GCN(in_channels, self.emb_size, agg="sum", residual=False)])
        for _ in range(args.num_layer - 1):
            self.gcn_layers.append(GCN(self.emb_size, self.emb_size, agg="sum"))
        self.pool = SumPooling() 
        
    def forward(self, ob):
        ob_g = [o['g'] for o in ob]
        ob_att = [o['att'] for o in ob]

        # create attachment point mask as one-hot
        for i, x_g in enumerate(ob_g):
            att_onehot = F.one_hot(torch.LongTensor(ob_att[i]), 
                        num_classes=x_g.number_of_nodes()).sum(0)
            ob_g[i].ndata['att_mask'] = att_onehot.bool()

        g = deepcopy(dgl.batch(ob_g)).to(self.device)
        
        g.ndata['x'] = self.emb_linear(g.ndata['x'])

        for i, conv in enumerate(self.gcn_layers):
            h = conv(g)
            g.ndata['x'] = h
        
        emb_node = g.ndata['x']

        ## Get graph embedding
        emb_graph = self.pool(g, g.ndata['x'])
        
        return g, emb_node, emb_graph


class GCNActorCritic(nn.Module):
    def __init__(self, env, args, vocab, predictor=False):
        super().__init__()

        self.embed = GCNEmbed(args, vocab['ATOM'])
        self.pi = SFSPolicy(env, args, vocab['FRAG_MOL'])
        self.q1 = GCNQFunction(args)
        self.q2 = GCNQFunction(args, override_seed=True)
        if predictor:
            self.p = GCNPredictor(args, vocab['ATOM'])
