import torch
from torch import nn

from .torch_interface import ExpertGather, ExpertScatter

import torch.nn.functional as F

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

class TorchGatherEinsum(nn.Module):
    def __init__(self, E: int, I: int, J: int):
        super().__init__()

        self.E, self.I, self.J = E, I, J
        self.W = nn.Parameter(torch.rand(E, I, J))

    def forward(self, X, ind):
        B, T, _ = X.shape
        _, E, K = ind.shape
        X = X.unsqueeze(1).expand(-1, self.E, -1, -1)
        
        X_gathered = torch.gather(X, dim=2, index=ind[...,None])
        Y = torch.einsum('beki, eij->bekj', X_gathered, self.W)
        #Y_gathered = torch.gather(Y, dim=2, index=ind.unsqueeze(-1).expand(-1,-1,-1,self.J))
        return Y
    
    def match_linear(self, egs_gather: ExpertGather):
        self.W = nn.Parameter(egs_gather.W)

class TorchReshapedGatherEinsum(nn.Module):
    def __init__(self, E: int, I: int, J: int):
        super().__init__()

        self.E, self.I, self.J = E, I, J
        self.W = nn.Parameter(torch.rand(E, I, J))

    def forward(self, X, ind):
        B, T, I = X.shape
        _, E, K = ind.shape

        index=ind.reshape(B, E*K)[...,None].expand(-1,-1,I)
        X_gathered = torch.gather(X, dim=1, index=index).reshape(B, E, K, I)
        Y = torch.einsum('beki, eij->bekj', X_gathered, self.W)
        return Y
    
    def match_linear(self, egs_gather: ExpertGather):
        self.W = nn.Parameter(egs_gather.W)

class TorchReshapedEmbeddingGatherEinsum(nn.Module):
    def __init__(self, E: int, I: int, J: int):
        super().__init__()

        self.E, self.I, self.J = E, I, J
        self.W = nn.Parameter(torch.rand(E, I, J))

    def forward(self, X, ind):
        B, T, I = X.shape
        _, E, K = ind.shape
        # Create an offset so that each batch's indices point to the correct part of the flattened X.
        offset = torch.arange(B, device=X.device).view(B, 1, 1) * T
        # Flatten X from (B, T, I) to (B*T, I)
        X_flat = X.reshape(B * T, I)
        # Now, adjust indices to index into the flattened X.
        X_gathered = F.embedding(ind + offset, X_flat)  # Shape: (B, E, K, I)
        Y = torch.einsum('beki, eij->bekj', X_gathered, self.W)
        return Y
    
    def match_linear(self, egs_gather: ExpertGather):
        self.W = nn.Parameter(egs_gather.W)



class FusedGatherEinsumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, ind, W, offset):
        # X: (B, T, I)
        # ind: (E, K) or (B, E, K) - adjust as needed; here we assume (B, E, K)
        B, T, I = X.shape
        _, E, K = ind.shape
        
        # Flatten X for embedding
        X_flat = X.reshape(B * T, I)
        # Perform embedding lookup with adjusted indices
        gathered = F.embedding(ind + offset, X_flat)  # shape: (B, E, K, I)
        # Perform einsum to combine gathered values with weight tensor W of shape (E, I, J)
        Y = torch.einsum('beki, eij -> bekj', gathered, W)
        
        # Save only necessary values for backward
        ctx.save_for_backward(X, ind+offset, W)
        ctx.shape_info = (B, T, I, E, K)
        return Y

    @staticmethod
    def backward(ctx, grad_output):
        X, ind_offset, W = ctx.saved_tensors
        B, T, I, E, K = ctx.shape_info
        
        # Recompute X_flat and gathered tensor
        X_flat = X.reshape(B * T, I)
        gathered = F.embedding(ind_offset, X_flat)  # shape: (B, E, K, I)
        
        # Compute gradient for gathered via the einsum backward:
        # grad_gathered corresponds to the gradient w.r.t. the output of F.embedding
        grad_gathered = torch.einsum('bekj, eij -> beki', grad_output, W)
        
        # Now, compute grad_X by scattering grad_gathered back to the flattened X.
        grad_X_flat = torch.zeros_like(X_flat)
        # Prepare indices for scatter_add: shape (B, E, K) -> (B*E*K, 1) then expand to (B*E*K, I)
        index = (ind_offset).reshape(-1, 1).expand(-1, I)
        grad_X_flat = grad_X_flat.scatter_add(0, index, grad_gathered.reshape(-1, I))
        grad_X = grad_X_flat.reshape(B, T, I)
        
        # Compute gradient for W using the gathered tensor and grad_output
        grad_W = torch.einsum('beki, bekj -> eij', gathered, grad_output)
        
        # No gradient for ind as it is an integer tensor
        return grad_X, None, grad_W, None

class TorchFusedReshapedEmbeddingGatherEinsum(nn.Module):
    def __init__(self, E: int, I: int, J: int, B: int, T: int):
        super().__init__()

        self.E, self.I, self.J = E, I, J
        self.W = nn.Parameter(torch.rand(E, I, J))

        # Create offset for embedding lookup
        self.offset = torch.arange(B, device='cuda').view(B, 1, 1) * T

    def forward(self, X, ind):
        return FusedGatherEinsumFunction.apply(X, ind ,self.W, self.offset)
    
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

        X_prescatter = torch.einsum('bekj, eji->beki', Y, self.W)

        I = X_prescatter.shape[-1]
        scattered = torch.zeros(B, T, I, device=Y.device, dtype=Y.dtype)
        Ind = Ind[..., None].expand(-1,-1,-1,I)
        scattered.scatter_add_(1, Ind.reshape(B, E*K, I), X_prescatter.reshape(B, E*K, I))
        return scattered
        
    def match_linear(self, egs_gather: ExpertGather):
        self.W = nn.Parameter(egs_gather.W)