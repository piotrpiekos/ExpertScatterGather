import torch
from torch import nn

from .torch_interface import ExpertGather, ExpertScatter


class TorchLinearGather(nn.Module):
    def __init__(self, E: int, I: int, J: int):
        super().__init__()

        self.E, self.I, self.J = E, I, J
        self.head_projection = torch.nn.Linear(I, E * J, bias=False, dtype=torch.float32)

    def forward(self, X, ind):
        B, T, _ = X.shape
        Y = self.head_projection(X).reshape(B, T, self.E, self.J).transpose(1,2)
        Y_gathered = torch.gather(Y, dim=2, index=ind.unsqueeze(-1).expand(-1,-1,-1,self.J))
        return Y_gathered
    
    def match_linear(self, egs_gather: ExpertGather):
        self.head_projection.weight = nn.Parameter(egs_gather.W.reshape(self.I, self.E*self.J))


class TorchEinsumGather(nn.Module):
    def __init__(self, E: int, I: int, J: int):
        super().__init__()

        self.E, self.I, self.J = E, I, J
        self.W = nn.Parameter(torch.rand(E, I, J))

    def forward(self, X, ind):
        B, T, _ = X.shape
        X = X.unsqueeze(1).expand(-1, self.E, -1, -1)
        Y = torch.einsum('beti, eij->betj', X, self.W)
        Y_gathered = torch.gather(Y, dim=2, index=ind.unsqueeze(-1).expand(-1,-1,-1,self.J))
        return Y_gathered
    
    def match_linear(self, egs_gather: ExpertGather):
        self.W = nn.Parameter(egs_gather.W)

class TorchEinsumScatter(nn.Module):
    def __init__(self, E: int, J: int, I: int):
        super().__init__()

        self.E, self.J, self.I = E, J, I
        self.W = nn.Parameter(torch.rand(E, J, I))

    def forward(self, Y, Ind, T):
        B, E, K, J = Y.shape
        # Ind shape [B, E, K]

        scattered = torch.zeros(B, E, T, J, device=Y.device, dtype=Y.dtype)
        Ind = Ind[..., None].expand(-1,-1,-1,J)
        scattered = torch.scatter_add(scattered, dim=2, index=Ind, src=Y)
        scattered_i = torch.einsum('betj, eji->beti', scattered, self.W)
        out = scattered_i.sum(dim=1) # [B, T, I]
        return out
        
    def match_linear(self, egs_gather: ExpertGather):
        self.W = nn.Parameter(egs_gather.W)