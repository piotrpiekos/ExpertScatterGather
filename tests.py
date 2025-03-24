import torch
from torch import nn

from expert_gather import SelectiveLinearLayer

DEVICE = 'cuda'

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
    
    def match_linear(self, egs_gather: SelectiveLinearLayer):
        self.head_projection.weight = nn.Parameter(egs_gather.W.reshape(I, E*J))


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
    
    def match_linear(self, egs_gather: SelectiveLinearLayer):
        self.W = nn.Parameter(egs_gather.W)
    

def random_test(B, T, K, E, I, J):
    X = torch.randn((B, T, I), device=DEVICE, dtype=torch.float32, requires_grad=True)
    W = torch.randn((E, I, J), device=DEVICE, dtype=torch.float32, requires_grad=True)
    Ind = torch.randint(0, T-1, (B, E, K), device=DEVICE, dtype=torch.int16, requires_grad=False)
    Ind64 = Ind.to(torch.int64)

    egs_expert_gather = SelectiveLinearLayer(E, I, J).to(DEVICE)
    torch_expert_gather = TorchEinsumGather(E, I, J).to(DEVICE)
    # match the weights of both
    torch_expert_gather.W = nn.Parameter(egs_expert_gather.W)

    # Calculate outputs for egs
    Y_egs = egs_expert_gather(X, Ind)
    Y_egs.sum().backward()
    X_grad_egs = X.grad.clone()
    W_grad_egs = egs_expert_gather.W.grad.clone()

    X.grad.zero_()
    egs_expert_gather.W.grad.zero_()
    
    # Calculate outputs for torch
    Y_torch = torch_expert_gather(X, Ind64)
    Y_torch.sum().backward()
    X_grad_torch = X.grad.clone()
    W_grad_torch = torch_expert_gather.W.grad.clone()

    print('max difference Y: ', (Y_egs - Y_torch).abs().max())
    print('max difference gradX: ', (X_grad_egs - X_grad_torch).abs().max())
    print('max difference gradW: ', (W_grad_egs - W_grad_torch).abs().max())
    assert torch.isclose(Y_egs, Y_torch, atol=1e-3).all()
    assert torch.isclose(X_grad_egs, X_grad_torch, atol=1e-3).all()
    assert torch.isclose(W_grad_egs, W_grad_torch, atol=1e-3).all()

    




B, T, I = 64, 512, 1024   # For X: [B, T, I]
E, J = 16, 64          # For W (grad_W): [E, I, J]
K = 64
random_test(B, T, K, E, I, J)