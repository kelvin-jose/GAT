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
        pass

    def init_params(self):
        torch.nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        torch.nn.init.xavier_uniform_(self.source_scorer)
        torch.nn.init.xavier_uniform_(self.target_scorer)