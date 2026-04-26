import torch as th
from torch import nn
from torch.nn import functional as F

'''
Training test file for the deep model transferrability studies
1. added norm weight in the backward pass
2. has been tested on wide and deep networks --- satisfactory performances

Currently treated as the correct model 
'''

class EmbedLayerUpdate(nn.Module):
    '''
    this class updates embeded layer one time
    '''
    def __init__(self,p, T):
        super(EmbedLayerUpdate, self).__init__()
        '''
        p: embedding size
        T: number of passes
        '''
        self.p = p
        self.T = T

        #inputs
        self.inp_f = nn.Linear(3,p)
        self.inp_f_1 = nn.Linear(p,p)

        self.inp_b = nn.Linear(2,p)
        self.inp_b_1 = nn.Linear(p,p)
        self.inp_b2 = nn.Linear(2*p,p)
        self.inp_b2_2 = nn.Linear(p,p)
        #self.inp_b = nn.Linear(p, p)

        # for bound features of the node
        self.fc1 = nn.Linear(7, p)      
        self.fc1_1 = nn.Linear(p, p)      
        #self.fc1_2 = nn.Linear(p, p)      
        #self.fc2 = nn.Linear(3*p, p)
        #self.fc2_1 = nn.Linear(p, p)

        # for sum of neighbour embeddings vectors
        self.fc3 = nn.Linear(2*p, p)     
        self.fc3_2 = nn.Linear(p, p)     
        # for weights of edges connecting to the node. 
        # edge updates
        self.fc4 = nn.Linear(2*p,p)
        self.fc4_2 = nn.Linear(p,p)

        # outputs
        self.out1 = nn.Linear(4, p)
        self.out2 = nn.Linear(2*p, p)
        self.out3 = nn.Linear(p, p)
        
        #backwards
        self.bc1 = nn.Linear(7, p)
        self.bc1_1 = nn.Linear(p, p)
        self.bc1_2 = nn.Linear(p, p)
        self.bc2 = nn.Linear(3*p, p)
        self.bc2_1 = nn.Linear(p, p)
        self.bc3 = nn.Linear(2*p, p)
        self.bc3_1 = nn.Linear(p, p)
        self.bc4 = nn.Linear(2*p, p)
        self.bc4_1 = nn.Linear(p, p)


    def forward(
        self,
        lower_bounds_all,
        upper_bounds_all,
        dual_vars,
        primals,
        primal_inputs,
        layers,
        mu,
        include_output_embedding: bool = True,
    ):
        # Lightweight path: build node embeddings directly from bound features
        # (BaBSR/UPB-style), then score unstable nodes with the learned heads.
        batch_size = len(lower_bounds_all[0])
        p = self.p
        device = lower_bounds_all[0].device
        dtype = lower_bounds_all[0].dtype

        inp_mid = primal_inputs
        inp_feat = th.cat(
            [
                lower_bounds_all[0].reshape(-1, 1),
                inp_mid.reshape(-1, 1),
                upper_bounds_all[0].reshape(-1, 1),
            ],
            dim=1,
        )
        mu[0] = self.inp_f_1(F.relu(self.inp_f(inp_feat))).reshape(mu[0].size())

        if include_output_embedding:
            hidden_lbs = [lb.reshape(batch_size, -1) for lb in lower_bounds_all[1:-1]]
            hidden_ubs = [ub.reshape(batch_size, -1) for ub in upper_bounds_all[1:-1]]
        else:
            hidden_lbs = [lb.reshape(batch_size, -1) for lb in lower_bounds_all[1:]]
            hidden_ubs = [ub.reshape(batch_size, -1) for ub in upper_bounds_all[1:]]
        if len(hidden_lbs) == 0:
            return mu

        n_per_layer = [x.size(1) for x in hidden_lbs]
        lb_cat = th.cat(hidden_lbs, dim=1)
        ub_cat = th.cat(hidden_ubs, dim=1)

        ratio_0, _, beta, ambi_mask = compute_ratio(lb_cat, ub_cat)
        zero_feat = th.zeros_like(lb_cat)
        node_feat = th.stack(
            [beta, lb_cat, ub_cat, zero_feat, zero_feat, zero_feat, zero_feat], dim=-1
        )

        node_feat_flat = node_feat.reshape(-1, 7)
        relax = self.fc1_1(F.relu(self.fc1(node_feat_flat))).reshape(batch_size, -1, p)
        relax = relax * ambi_mask.unsqueeze(-1)

        nb_input = th.zeros(batch_size, lb_cat.size(1), 2 * p, device=device, dtype=dtype)
        nb_embed = self.fc3_2(F.relu(self.fc3(nb_input.reshape(-1, 2 * p)))).reshape(
            batch_size, -1, p
        )
        nb_embed = nb_embed * ratio_0.unsqueeze(-1)

        mu_hidden = self.fc4_2(
            F.relu(self.fc4(th.cat([relax, nb_embed], dim=-1).reshape(-1, 2 * p)))
        ).reshape(batch_size, -1, p)
        mu_hidden = mu_hidden * ambi_mask.unsqueeze(-1)

        for i, part in enumerate(th.split(mu_hidden, n_per_layer, dim=1), start=1):
            mu[i] = part.reshape(mu[i].size())

        if include_output_embedding:
            out_lb = lower_bounds_all[-1].reshape(-1)
            out_ub = upper_bounds_all[-1].reshape(-1)
            out_feat = th.cat(
                [
                    out_lb.unsqueeze(-1),
                    out_ub.unsqueeze(-1),
                    th.zeros_like(out_lb).unsqueeze(-1),
                    th.zeros_like(out_lb).unsqueeze(-1),
                ],
                dim=1,
            )
            out_relax = F.relu(self.out1(out_feat))
            out_nb = th.zeros(out_relax.size(0), p, device=device, dtype=dtype)
            mu[-1] = self.out3(F.relu(self.out2(th.cat([out_relax, out_nb], dim=1)))).reshape(
                mu[-1].size()
            )
        return mu





class EmbedUpdates(nn.Module):
    '''
    this class updates embeding vectors from t=1 and t=T
    '''

    def __init__(self, T, p):
        '''
        p_list contains the input and output dimensions of embedding vectors for all layers
        len(p_list) = T+1
        '''
        super(EmbedUpdates, self).__init__()
        self.T = T
        self.p = p
        #self.p_list = p_list
        #assert len(p_list)==T+1
        #self.updates = [EmbedLayerUpdate(p_list[i], p_list[i+1]) for i in range(0,T)]
        self.update = EmbedLayerUpdate(p,T)


    def forward(
        self,
        lower_bounds_all,
        upper_bounds_all,
        dual_vars,
        primals,
        primal_inputs,
        layers,
        include_output_embedding: bool = True,
    ):
        #mu = lower_bounds_all[0].new_full((len(lower_bounds_all[0]), self.p), fill_value=0.0)
        mu = init_mu(lower_bounds_all, self.p)
        mu = self.update(
            lower_bounds_all,
            upper_bounds_all,
            dual_vars,
            primals,
            primal_inputs,
            layers,
            mu,
            include_output_embedding=include_output_embedding,
        )
        return mu



class ComputeFinalScore(nn.Module):
    '''
    this class computes a final score for each node

    p: the dimension of embedding vectors at the final stage
    '''
    def __init__(self, p):
        super(ComputeFinalScore,self).__init__()
        self.p = p
        #self.fnode = nn.Linear(p,1)
        self.fnode = nn.Linear(p,p)
        self.fscore = nn.Linear(p, 1)
    
    def utils(self, mu):
        scores = []
        for layer in mu[1:-1]:
            scores_current_layer = mu[0].new_full((layer.size()[0],), fill_value=0.0) 
            scores.append(scores_current_layer)
        return scores


    def forward(self, mu, masks, include_output_embedding: bool = True):
        scores = []
        mu_hidden = th.cat(mu[1:-1] if include_output_embedding else mu[1:], dim=1)
        for batch_idx in range(mu_hidden.size(0)):
            valid = masks[batch_idx].bool()
            mu_valid = mu_hidden[batch_idx][valid]
            temp = self.fnode(mu_valid)
            score = self.fscore(F.relu(temp))
            scores.append(score.view(-1))
        #scores = nn.utils.rnn.pad_sequence(scores, batch_first=True)

        #for layer_idx in range(1, len(mu)-1):
        #    temp = self.fnode(mu[layer_idx])
        #    layer_score = self.fscore(F.relu(temp))
        #    scores[layer_idx-1] = layer_score
        
        #import db; pdb.set_trace()
        # scores = [i.squeeze(1) for i in scores]
        # reshape into 1 dimension
        # scores = th.cat([i for i in scores],0)
        # remove all invalid options
        # scores = scores[mask.nonzero()].view(-1)

        #scores = th.cat([i for i in scores],0)
        #scores = scores[mask.nonzero().view(-1)]
        #scores = scores/max(scores)
        #scores = self.fscore_3(F.relu(self.fscore_2(scores)))
        #transform into relative number
        return scores


class GraphNet(nn.Module):
    def __init__(self, T, p):
        super(GraphNet,self).__init__()
        self.EmbedUpdates = EmbedUpdates(T, p)
        self.ComputeFinalScore = ComputeFinalScore(p)

    def forward(
        self,
        lower_bounds_all,
        upper_bounds_all,
        dual_vars,
        primals,
        primal_inputs,
        layers,
        masks,
        include_output_embedding: bool = True,
    ):
        mu = self.EmbedUpdates(
            lower_bounds_all,
            upper_bounds_all,
            dual_vars,
            primals,
            primal_inputs,
            layers,
            include_output_embedding=include_output_embedding,
        )
        scores = self.ComputeFinalScore(mu, masks, include_output_embedding=include_output_embedding)

        return scores



def init_mu(lower_bounds_all, p):
    
    mu = []
    batch_size = len(lower_bounds_all[0])
    for i in lower_bounds_all:
        required_size = i[0].view(-1).size()
        mus_current_layer = lower_bounds_all[0].new_full((batch_size,required_size[0],p), fill_value=0.0) 
        mu.append(mus_current_layer)
    
    return mu


def compute_ratio(lower_bound, upper_bound):
    lower_temp = lower_bound - F.relu(lower_bound)
    upper_temp = F.relu(upper_bound)
    slope_ratio0 = upper_temp/(upper_temp-lower_temp)
    intercept = -1*lower_temp*slope_ratio0
    ambi_mask = (intercept>0).float()
    slope_ratio1 = (1 - 2*(slope_ratio0*ambi_mask))*ambi_mask + slope_ratio0
    #if (lower >= 0):
    #    decision = [1,0]
    #elif (upper <=0):
    #    decision = [0, 0]
    #else: 
    #    temp = upper/(upper - lower)
    #    decision = [temp, -temp*lower]
    #return decision
    return slope_ratio0, slope_ratio1, intercept, ambi_mask


