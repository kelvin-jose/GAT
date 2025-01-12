import torch

class GATLayer(torch.nn.Module):
    def __init__(self, fin, fout, num_heads, concat=True):
        super().__init__()
        self.num_heads = num_heads
        self.fout = fout
        self.W = torch.nn.Linear(fin, num_heads * fout, bias=False)
        self.source_scorer = torch.nn.Parameter(torch.Tensor(num_heads, 1, fout))
        self.target_scorer = torch.nn.Parameter(torch.Tensor(num_heads, 1, fout))
        self.l_relu = torch.nn.ELU(0.2)
        self.concat = concat
        self.drop1 = torch.nn.Dropout(0.6)
        self.drop2 = torch.nn.Dropout(0.6)

    def forward(self):
        x = self.drop1(x)
        x = self.W(x).view(self.num_heads, -1, self.fout) # [num_heads, nodes, fout]
        source_scores = (x * self.source_scorer).sum(-1)
        target_scores = (x * self.target_scorer).sum(-1)
        select_feats, select_scores = self.select_source_target_feats(source_scores, target_scores, x)
        scores = self.l_relu(select_scores.sum(axis=-1))


    def init_params(self):
        torch.nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        torch.nn.init.xavier_uniform_(self.source_scorer)
        torch.nn.init.xavier_uniform_(self.target_scorer)

    
    def select_source_target_feats(self, source_scores, target_scores, features, edge_index):
        selected_scores = torch.zeros(self.num_heads, edge_index.shape[0], edge_index.shape[1])
        selected_features = torch.index_select(features, 1, torch.tensor(edge_index[:, 0]))
        selected_scores[:, :, 0] = torch.index_select(source_scores, 1, torch.tensor(edge_index[:, 0]))
        selected_scores[:, :, 1] = torch.index_select(target_scores, 1, torch.tensor(edge_index[:, 1]))
        return selected_features, selected_scores
    
    def softmax_sum(self, scores, features, x_shape, edge_index):        
        scores_target = torch.zeros(self.num_heads, x_shape[1])
        feat_target = torch.zeros(x_shape)
        scores_exp = scores.exp()
        target_edge_indices = torch.tensor(edge_index[:, 1])
        sm_denom = torch.scatter_add(scores_target, 
                                     dim = 1, 
                                     index = target_edge_indices.unsqueeze(0).repeat(self.num_heads, 1), 
                                     src = scores_exp)
        sm_denom_expanded = torch.index_select(sm_denom, 1, target_edge_indices)
        att_scores = scores_exp / sm_denom_expanded
        att_scores = self.drop2(att_scores)
        weighted_feats = features * att_scores.unsqueeze(-1)
        weighted_feats_sum = torch.scatter_add(feat_target, 
                                               dim = 1, 
                                               index = target_edge_indices.unsqueeze(-1).repeat(self.num_heads, 1, self.fout), 
                                               src = weighted_feats)
        return weighted_feats_sum, att_scores