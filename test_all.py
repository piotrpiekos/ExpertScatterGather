import torch
from torch import nn

from egs import ExpertGather, ExpertScatter
from egs.baselines import TorchLinearGather, TorchEinsumGather, TorchEinsumScatter, TorchReshapedEmbeddingGatherEinsum, TorchFusedReshapedEmbeddingGatherEinsum

DEVICE = 'cuda'

def compare(T1, T2, name, atol=1e-3):
    print(f'max difference {name}: ', (T1 - T2).abs().max())
    assert torch.isclose(T1, T2, atol=atol).all()

def shape_test_gather(B, T, K, E, I, J):
    X = torch.randn((B, T, I), device=DEVICE, dtype=torch.float32, requires_grad=True)
    W = torch.randn((E, I, J), device=DEVICE, dtype=torch.float32, requires_grad=True)
    Ind = torch.randint(0, T, (B, E, K), device=DEVICE, dtype=torch.int16, requires_grad=False)
    Ind64 = Ind.to(torch.int64)

    egs_expert_gather = ExpertGather(E, I, J).to(DEVICE)
    #torch_expert_gather = TorchEinsumGather(E, I, J).to(DEVICE)
    # torch_expert_gather = TorchReshapedEmbeddingGatherEinsum(E, I, J).to(DEVICE)
    torch_expert_gather = TorchFusedReshapedEmbeddingGatherEinsum(E, I, J).to(DEVICE)
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

