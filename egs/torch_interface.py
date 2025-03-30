import math
from packaging import version

import torch
from torch import nn

#from .triton_kernels import create_kernels, expert_gather_backward_call, expert_gather_forward_call, expert_scatter_backward_call, expert_scatter_forward_call
from .triton_kernels import ExpertGatherFunction, ExpertScatterFunction, create_kernels


class ExpertGather(nn.Module):
    def __init__(self, E: int, I: int, J: int, bias: bool=False):
        super().__init__()
        self.E, self.I, self.J = E, I, J
        self.W = nn.Parameter(torch.Tensor(E, I, J))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(E, 1, J))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        if (version.parse(torch.__version__) >= version.parse("2.2.0") and
                torch.cuda.get_device_properties(0).major == 7 and
                torch.cuda.get_device_properties(0).minor < 5 and
                torch.is_autocast_enabled()):
            print("------------------------------- ERROR -------------------------------")
            print("ERROR: PyTorch >= 2.2 with AMP is be broken on Volta GPUs.")
            print("Triton kernels returns zeroes only. Please downgrade to 2.1 series.")
            print("Alternatively, disable mixed precision training")
            print("See: https://github.com/pytorch/pytorch/issues/127157")
            print("---------------------------------------------------------------------")
            raise RuntimeError("PyTorch >= 2.2 Triton with AMP is to be broken on Volta GPUs.")
        create_kernels()


    def reset_parameters(self):
        bound = 1 / math.sqrt(self.I)
        nn.init.uniform_(self.W, -bound, bound)

        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)
            

    def forward(self, x: torch.Tensor, Ind: torch.Tensor):
        """
        Args: 
            x: torch.Tensor, shape [B, T, I], 
            Ind: torch.Tensor of integers from 0 to T - 1, shape # [B, E, K]
        Return:
            x gathered on indices Ind and transformed by internal linear layer: torch.Tensor of shape [B, E, K, J]
        """

        y = ExpertGatherFunction.apply(x, self.W, Ind)
        if self.bias:
            y = y + self.bias 
        return y 

    def to(self, device, *args, **kwargs):
        self.W = nn.Parameter(self.W.to(device, *args, **kwargs))
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.to(device, *args, **kwargs))
        return self
    

class ExpertScatter(nn.Module):
    def __init__(self, E: int, J: int, I: int, bias: bool =False):
        super().__init__()
        self.E, self.J, self.I = E, J, I
        self.W = nn.Parameter(torch.Tensor(E, J, I))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(E, 1, I))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        if (version.parse(torch.__version__) >= version.parse("2.2.0") and
                torch.cuda.get_device_properties(0).major == 7 and
                torch.cuda.get_device_properties(0).minor < 5 and
                torch.is_autocast_enabled()):
            print("------------------------------- ERROR -------------------------------")
            print("ERROR: PyTorch >= 2.2 with AMP is be broken on Volta GPUs.")
            print("Triton kernels returns zeroes only. Please downgrade to 2.1 series.")
            print("Alternatively, disable mixed precision training")
            print("See: https://github.com/pytorch/pytorch/issues/127157")
            print("---------------------------------------------------------------------")
            raise RuntimeError("PyTorch >= 2.2 Triton with AMP is to be broken on Volta GPUs.")
        create_kernels()


    def reset_parameters(self):
        bound = 1 / math.sqrt(self.I)
        nn.init.uniform_(self.W, -bound, bound)

        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)
            

    def forward(self, Y: torch.Tensor, Ind: torch.Tensor, T: int):
        """
        Args: 
            Y: torch.Tensor, shape [B, E, K, J], 
            Ind: torch.Tensor of integers from 0 to T - 1, shape # [B, E, K]
            T: sequence length of output 
        Return:
            Y scathered on indices Ind and transformed by internal linear layer: torch.Tensor of shape [B, T, I]
        """

        y = ExpertScatterFunction.apply(Y, self.W, Ind, T)
        if self.bias:
            y = y + self.bias
        return y 

    def to(self, device, *args, **kwargs):
        self.W = nn.Parameter(self.W.to(device, *args, **kwargs))
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.to(device, *args, **kwargs))
        return self