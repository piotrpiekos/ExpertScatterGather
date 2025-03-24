
import torch
from torch import nn

import triton
import triton.language as tl

import matplotlib

import math

from expert_gather import ExpertGather
class TorchEinsumGather(nn.Module):
    def __init__(self, E: int, I: int, J: int):
        super().__init__()

        self.E, self.I, self.J = E, I, J
        self.W = nn.Parameter(torch.rand(E, I, J))

    def forward(self, X, ind):
        B, T, _ = X.shape
        X = X.unsqueeze(1).expand(-1, self.E, -1, -1)

        # Y = torch.einsum('beti,eij->betj', X, self.W)
        # Y_gathered = torch.gather(Y, dim=2, index=ind.unsqueeze(-1).expand(-1,-1,-1,self.J))


        X_gathered = torch.gather(X, dim=2, index=ind.unsqueeze(-1).expand(-1,-1,-1,self.I))
        Y_gathered = torch.einsum('beki, eij->bekj', X_gathered, self.W)
        #Y_gathered = torch.gather(Y, dim=2, index=ind.unsqueeze(-1).expand(-1,-1,-1,self.J))
        return Y_gathered

    def match_linear(self, egs_gather: ExpertGather):
        self.W = nn.Parameter(egs_gather.W)
    

class TorchLinearGather(nn.Module):
    def __init__(self, E: int, I: int, J: int):
        super().__init__()

        self.E, self.I, self.J = E, I, J
        self.W = nn.Linear(I, E * J, bias=False)

    def forward(self, X, ind):
        B, T, I = X.shape
        Y = self.W(X).reshape(B, self.E, T, J)
        Y_gathered = torch.gather(Y, dim=2, index=ind.unsqueeze(-1).expand(-1,-1,-1,self.J))
        return Y_gathered

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

E = 32
T = 128
K = 32
B = 64
J = 64
I = 256

ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'
# We will compare our Triton implementation vs torch.matmul (which uses the ref_lib).
# Note: torch.matmul does not support FP8, so we only consider fp16 here.
lim=5
configs = [
    triton.testing.Benchmark(
        x_names=["T"],
        #x_vals=[(128 * i, 128 * j, 128 * k) for i in range(2, lim) for j in range(2,lim) for k in range(2,lim)],
        #x_vals = [(i, B, J, I) for i in range(2, 6)],
        x_vals=[128 * (i+1) for i in range(10)],
        line_arg="provider",
        line_vals=["torch", "esg"],
        line_names=["Torch", "esg"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="ms",
        plot_name="time needed for a forward and a backward pass",
        args={
            "fp8_inputs": False,
            "B": B, "E": E, "J": J, "I": I, "K": K
            },
    )
]

@triton.testing.perf_report(configs)
def benchmark(T, B, E, J, I, K, provider, fp8_inputs):
    quantiles = [0.5, 0.2, 0.8]

    DEVICE = "cuda"  # or "cpu", but ensure Triton is set up for your device

    # Create batched inputs.
    X = torch.randn((B, T, I), device=DEVICE, dtype=torch.float32, requires_grad=True)
    W = torch.randn((E, I, J), device=DEVICE, dtype=torch.float32, requires_grad=True)
    kernel_head_projection = ExpertGather(E, I, J).to(DEVICE)
    torch_head_projection = TorchEinsumGather(E, I, J).to(DEVICE)
    torch_head_projection.W = nn.Parameter(kernel_head_projection.W)

    Ind = torch.randint(0, T-1, (B, E, K), device=DEVICE, dtype=torch.int16, requires_grad=False)
    Ind64 = Ind.to(torch.int64)
    if provider == 'torch':
        def run():
            # forward + backward using torch.matmul
            Y_torch = torch_head_projection(X, Ind64)
            l = Y_torch.sum()
            l.backward()
        ms, min_ms, max_ms = triton.testing.do_bench(run, quantiles=quantiles)
    elif provider == "esg":
        def run():
            # forjward + backward using our custom Triton function
            Y_esg = kernel_head_projection(X, Ind)
            #Y_triton = ExpertMatMul.apply(X, W, Ind, "none")
            l = Y_esg.sum()
            l.backward()
        ms, min_ms, max_ms = triton.testing.do_bench(run, quantiles=quantiles)
    else:
        raise ValueError("Unknown provider")

    # For a GEMM, forward flops = 2*M*N*K.
    # Backward pass computes two GEMMs, so add 2*2*M*N*K => total flops ~ 6*M*N*K.

    return ms, max_ms, min_ms
benchmark.run(show_plots=True, print_data=True)

print('='*20)
print('MEMORY')
print('='*20)

# Reference library name for display.
ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'
# We will compare our Triton implementation vs torch.matmul (which uses the ref_lib).
# Note: torch.matmul does not support FP8, so we only consider fp16 here.
lim=5
configs = [
    triton.testing.Benchmark(
        x_names=["T"],
        #x_vals=[(128 * i, 128 * j, 128 * k) for i in range(2, lim) for j in range(2,lim) for k in range(2,lim)],
        #x_vals = [(i, B, J, I) for i in range(2, 6)],
        x_vals=[128 * (i+1) for i in range(10)],
        line_arg="provider",
        line_vals=["torch", "esg"],
        line_names=["Torch", "esg"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="ms",
        plot_name="time needed for a forward and a backward pass",
        args={
            "fp8_inputs": False,
            "B": B, "E": E, "J": J, "I": I, "K": K
            },
    )
]

@triton.testing.perf_report(configs)
def benchmark(T, B, E, J, I, K, provider, fp8_inputs):
    quantiles = [0.5, 0.2, 0.8]

    DEVICE = "cuda"  # or "cpu", but ensure Triton is set up for your device

    # Create batched inputs.
    X = torch.randn((B, T, I), device=DEVICE, dtype=torch.float32, requires_grad=True)
    W = torch.randn((E, I, J), device=DEVICE, dtype=torch.float32, requires_grad=True)
    kernel_head_projection = ExpertGather(E, I, J).to(DEVICE)
    torch_head_projection = TorchEinsumGather(E, I, J).to(DEVICE)
    torch_head_projection.W = nn.Parameter(kernel_head_projection.W)

    Ind = torch.randint(0, T-1, (B, E, K), device=DEVICE, dtype=torch.int16, requires_grad=False)
    Ind64 = Ind.to(torch.int64)
    torch.cuda.reset_peak_memory_stats()
    if provider == 'torch':
        def run():
            # forward + backward using torch.matmul
            Y_torch = torch_head_projection(X, Ind64)
            l = Y_torch.sum()
            l.backward()
        ms, min_ms, max_ms = triton.testing.do_bench(run, quantiles=quantiles)
    elif provider == "esg":
        def run():
            # forjward + backward using our custom Triton function
            Y_esg = kernel_head_projection(X, Ind)
            #Y_triton = ExpertMatMul.apply(X, W, Ind, "none")
            l = Y_esg.sum()
            l.backward()
        ms, min_ms, max_ms = triton.testing.do_bench(run, quantiles=quantiles)
    else:
        raise ValueError("Unknown provider")
    memory_used = torch.cuda.max_memory_allocated() / (1024 ** 2)
    # For a GEMM, forward flops = 2*M*N*K.
    # Backward pass computes two GEMMs, so add 2*2*M*N*K => total flops ~ 6*M*N*K.

    return memory_used, memory_used, memory_used
benchmark.run(show_plots=True, print_data=True)