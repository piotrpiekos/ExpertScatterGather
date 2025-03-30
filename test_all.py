import torch
from torch import nn

from expert_gather import ExpertGather, ExpertScatter

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
    
    def match_linear(self, egs_gather: ExpertGather):
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
    

def compare(T1, T2, name, atol=1e-3):
    print(f'max difference {name}: ', (T1 - T2).abs().max())
    assert torch.isclose(T1, T2, atol=atol).all()

def shape_test_gather(B, T, K, E, I, J):
    X = torch.randn((B, T, I), device=DEVICE, dtype=torch.float32, requires_grad=True)
    W = torch.randn((E, I, J), device=DEVICE, dtype=torch.float32, requires_grad=True)
    Ind = torch.randint(0, T, (B, E, K), device=DEVICE, dtype=torch.int16, requires_grad=False)
    Ind64 = Ind.to(torch.int64)

    egs_expert_gather = ExpertGather(E, I, J).to(DEVICE)
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

    return {
        'Y': (Y_egs, Y_torch),
        'X_grad': (X_grad_egs, X_grad_torch),
        'W_grad': (W_grad_egs, W_grad_torch)
    }

def shape_test_scatter(B, T, K, E, I, J):
    Y = torch.randn((B, E, K, J), device=DEVICE, dtype=torch.float32, requires_grad=True)
    W = torch.randn((E, J, I), device=DEVICE, dtype=torch.float32, requires_grad=True)
    Ind = torch.randint(0, T, (B, E, K), device=DEVICE, dtype=torch.int16, requires_grad=False)
    Ind64 = Ind.to(torch.int64)

    egs_expert_scatter = ExpertScatter(E, J, I).to(DEVICE)
    torch_expert_scatter = TorchEinsumScatter(E, J, I).to(DEVICE)
    # match the weights of both
    torch_expert_scatter.W = nn.Parameter(egs_expert_scatter.W)

    # Calculate outputs for egs
    X_egs = egs_expert_scatter(Y, Ind, T)
    X_egs.sum().backward()
    Y_grad_egs = Y.grad.clone()
    W_grad_egs = egs_expert_scatter.W.grad.clone()

    Y.grad.zero_()
    egs_expert_scatter.W.grad.zero_()
    
    # Calculate outputs for torch
    X_torch = torch_expert_scatter(Y, Ind64, T)
    X_torch.sum().backward()
    Y_grad_torch = Y.grad.clone()
    W_grad_torch = torch_expert_scatter.W.grad.clone()

    return {
        'X': (X_egs, X_torch),
        'Y_grad': (Y_grad_egs, Y_grad_torch),
        'W_grad': (W_grad_egs, W_grad_torch)
    }


### Random tests
NUM_RANDOM_TESTS = 5
MAX_DIM_SIZE = 128

def abstract_test(test_type, test_function):
    """
    General purpose test
    output_type=Y|X_grad|W_grad|X|Y_grad
    """
    assert test_type in {'Y', 'X_grad', 'W_grad', 'X', 'Y_grad'}, 'not recognized test type'
    assert test_function in {'gather', 'scatter'}

    shape_test_function = {
        'gather': shape_test_gather,
        'scatter': shape_test_scatter
    }[test_function]

    for _ in range(NUM_RANDOM_TESTS):
        B,T,K,E,I,J = torch.randint(1, MAX_DIM_SIZE, (6,))
        res_dict = shape_test_function(B,T,K,E,I,J)
        res_egs, res_torch = res_dict[test_type]
        compare(res_egs, res_torch, test_type)

def test_gather_forward():
    abstract_test('Y', 'gather')

def test_gather_gradX():
    abstract_test('X_grad', 'gather')

def test_gather_gradW():
    abstract_test('W_grad', 'gather')

def test_scatter_forward():
    abstract_test('X', 'scatter')

def test_scatter_gradY():
    abstract_test('Y_grad', 'scatter')

def test_scatter_gradW():
    abstract_test('W_grad', 'scatter')

