import torch
from torch import nn

import triton
import triton.language as tl

import math


from packaging import version


def get_dtype():
    if not torch.is_autocast_enabled():
        return torch.float32
    return torch.get_autocast_gpu_dtype()

def dtype_to_type_id(dtype: torch.dtype):
    if dtype == torch.float32:
        return 0
    elif dtype == torch.float16:
        return 1
    elif dtype == torch.bfloat16:
        return 2

    raise ValueError("Unknown dtype")

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip_cdna2():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_J': 64, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_B': 32, 'BLOCK_SIZE_J': 64, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_J': 32, 'BLOCK_SIZE_I': 128, 'GROUP_SIZE_B': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_J': 64, 'BLOCK_SIZE_I': 64, 'GROUP_SIZE_B': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_J': 64, 'BLOCK_SIZE_I': 256, 'GROUP_SIZE_B': 8}, num_stages=3, num_warps=8),
    ]

def get_hip_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_J': 256, 'BLOCK_SIZE_I': 16, 'GROUP_SIZE_B': 1, 'waves_per_eu': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_B': 256, 'BLOCK_SIZE_J': 256, 'BLOCK_SIZE_I': 16, 'GROUP_SIZE_B': 4, 'waves_per_eu': 2}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_J': 128, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 1, 'waves_per_eu': 2}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_J': 128, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 8, 'waves_per_eu': 3}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_B': 64, 'BLOCK_SIZE_J': 64, 'BLOCK_SIZE_I': 32, 'GROUP_SIZE_B': 1, 'waves_per_eu': 8}, num_warps=4, num_stages=2),
        ]

def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()
    
expert_gather_forward_call = None
expert_gather_backward_call = None

# create_kernels() overhead is necessary to operate in multigpu scenario and to utilize torch.compile
def create_kernels():
    global expert_gather_forward_call, expert_gather_backward_call

    if expert_gather_forward_call is not None:
        return


    @triton.autotune(
        configs=get_autotune_config(),
        key=['B', 'J', 'I', 'out_dtype_id', 'allow_tf32', 'dtype_id'], reset_to_zero = ['y_ptr'],
    )
    @triton.jit
    def expert_gather_forward(
            # Pointers for batched matrices:
            x_ptr, w_ptr, Ind_ptr, y_ptr,
            # Matrix dimensions:
            B, J, I,
            # Strides for X (shape: [B, T, I])
            stride_xb, stride_xT, stride_xi,
            # Strides for W (shape: [E, I, J])
            stride_we, stride_wi, stride_wj,
            # Stride for Ind (shape: [B, E, K])
            stride_Indb, stride_Inde, stride_Indk,
            # Strides for Y (shape: [B, E, K, J])
            stride_yb, stride_ye, stride_yk, stride_yj,
            # type parameters
            out_dtype_id: tl.constexpr, allow_tf32: tl.constexpr, dtype_id: tl.constexpr,
            # Meta-parameters:
            BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_J: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
            GROUP_SIZE_B: tl.constexpr,
    ):
        """
        Kernel to calculate expert gather
        Y[b,e,k,j] = \sum_{i=1}^{I} X\bigl[b,\operatorname{Ind}[b,e,k],i\bigr] \cdot W[e,i,j]
        Equivalent to expansion of X to multiple experts, selecting tokens from Ind and then application of the linear layer W.
        Based on https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
        """
        expert_id = tl.program_id(axis=0)
        pid = tl.program_id(axis=1)
        token_id = tl.program_id(axis=2)

        # Adjust pointers for the current batch.
        w_ptr = w_ptr + expert_id * stride_we
        y_ptr = y_ptr + expert_id * stride_ye + token_id * stride_yk
        Ind_ptr = Ind_ptr + expert_id * stride_Inde + token_id * stride_Indk
        # x_ptr is not adapted, because all experts take the same input

        # -----------------------------------------------------------
        # Map program ids to the block of Y to compute.
        num_pid_b = tl.cdiv(B, BLOCK_SIZE_B)
        num_pid_j = tl.cdiv(J, BLOCK_SIZE_J)
        num_pid_in_group = GROUP_SIZE_B * num_pid_j
        group_id = pid // num_pid_in_group
        first_pid_b = group_id * GROUP_SIZE_B
        group_size_b = min(num_pid_b - first_pid_b, GROUP_SIZE_B)
        pid_b = first_pid_b + ((pid % num_pid_in_group) % group_size_b)
        pid_j = (pid % num_pid_in_group) // group_size_b

        # -----------------------------------------------------------
        # Adjust pointers for the current in the head
        # one token and one expert for the kernel (multiple batches)
        offs_Ind_b = (pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)) % B
        Ind_ptr = Ind_ptr + offs_Ind_b * stride_Indb
        offs_xT = tl.load(Ind_ptr) # list of token ids in the sequence for given batch
        offs_xb = (pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)) % B
        offs_i = tl.arange(0, BLOCK_SIZE_I)
        x_ptrs = x_ptr + (offs_xb * stride_xb + offs_xT * stride_xT)[:, None] + offs_i[None, :] * stride_xi

        offs_wj = (pid_j * BLOCK_SIZE_J + tl.arange(0, BLOCK_SIZE_J)) % J
        w_ptrs = w_ptr + offs_i[:, None] * stride_wi + offs_wj[None, :] * stride_wj

        # -----------------------------------------------------------
        # Accumulate the product into a fp32 accumulator.
        accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_J), dtype=tl.float32)
        for i_idx in range(0, tl.cdiv(I, BLOCK_SIZE_I)):
            x = tl.load(x_ptrs, mask=offs_i[None, :] < I - i_idx * BLOCK_SIZE_I, other=0.0)
            w = tl.load(w_ptrs, mask=offs_i[:, None] < I - i_idx * BLOCK_SIZE_I, other=0.0)

            if dtype_id == 1:
                x = x.to(tl.float16)
                w = w.to(tl.float16)
            elif dtype_id == 2:
                x = x.to(tl.bfloat16)
                w = w.to(tl.bfloat16)
            accumulator = tl.dot(x, w, accumulator, allow_tf32=allow_tf32)

            x_ptrs += BLOCK_SIZE_I * stride_xi
            w_ptrs += BLOCK_SIZE_I * stride_wi

        if out_dtype_id == 1:
            y_out = accumulator.to(tl.float16)
        elif out_dtype_id == 2:
            y_out = accumulator.to(tl.bfloat16)
        else:
            y_out = accumulator

        # -----------------------------------------------------------
        # Write the computed block to Y.
        offs_yb = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        offs_yj = pid_j * BLOCK_SIZE_J + tl.arange(0, BLOCK_SIZE_J)
        y_ptrs = y_ptr + offs_yb[:, None] * stride_yb + offs_yj[None, :] * stride_yj
        y_mask = (offs_yb[:, None] < B) & (offs_yj[None, :] < J)
        tl.store(y_ptrs, y_out, mask=y_mask)

    @triton.autotune(
        configs=get_autotune_config(),
        key=['B', 'J', 'I', 'out_dtype_id', 'allow_tf32', 'dtype_id'], reset_to_zero = ['dX_ptr'],
    )
    @triton.jit
    def expert_gather_backward_dX_kernel(
            # Pointers for a batched tensor:
            dY_ptr, W_ptr, Ind_ptr, dX_ptr,
            # Dimensions:
            B, J, I,
            # Strides for Y (shape: [B, E, K, J])
            stride_dY_b, stride_dY_e, stride_dY_k, stride_dY_j,
            # Strides for W (shape: [E, I, J])
            stride_we, stride_wi, stride_wj,
            # Stride for Ind (shape: [B, E, K])
            stride_Indb, stride_Inde, stride_Indk,
            # Strides for X (shape: [B, T, I])
            stride_dX_b, stride_dX_T, stride_dX_i,
            # type parameters
            out_dtype_id: tl.constexpr, allow_tf32: tl.constexpr, dtype_id: tl.constexpr,
            # Meta-parameters:
            BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_J: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
            GROUP_SIZE_B: tl.constexpr,
    ):
        """
        Batched backward kernel to compute VJP w.r.t. X for expert_gather.
        \nabla_X[b,t,i] = \sum_{e=1}^{E} \sum_{k=1}^{K} \mathbf{1}{t = \operatorname{Ind}[b,e,k]} \cdot
                        \left( \sum_{j=1}^{J} \nabla Y[b,e,k,j] \cdot W[e,i,j] \right)
        """
        expert_id = tl.program_id(axis=0)
        pid = tl.program_id(axis=1)
        token_id = tl.program_id(axis=2)

        # Adjust pointers for the current batch.
        W_ptr = W_ptr + expert_id * stride_we
        dY_ptr = dY_ptr + expert_id * stride_dY_e + token_id * stride_dY_k
        Ind_ptr = Ind_ptr + expert_id * stride_Inde + token_id * stride_Indk

        # -----------------------------------------------------------
        # The rest is the same tiling strategy as before.
        # Just now the dimensions of the output are B and I
        num_pid_b = tl.cdiv(B, BLOCK_SIZE_B)
        num_pid_i = tl.cdiv(I, BLOCK_SIZE_I)
        num_pid_in_group = GROUP_SIZE_B * num_pid_i
        group_id = pid // num_pid_in_group
        first_pid_b = group_id * GROUP_SIZE_B
        group_size_b = min(num_pid_b - first_pid_b, GROUP_SIZE_B)
        pid_b = first_pid_b + ((pid % num_pid_in_group) % group_size_b)
        pid_i = (pid % num_pid_in_group) // group_size_b

        offs_yb = (pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)) % B
        offs_wi = (pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)) % I

        offs_j = tl.arange(0, BLOCK_SIZE_J)
        y_ptrs = dY_ptr + (offs_yb[:, None] * stride_dY_b + offs_j[None, :] * stride_dY_j)
        w_ptrs = W_ptr + (offs_j[:, None] * stride_wj + offs_wi[None, :] * stride_wi)

        accumulator = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_I), dtype=tl.float32)
        for j_idx in range(0, tl.cdiv(J, BLOCK_SIZE_J)):
            y = tl.load(y_ptrs, mask=offs_j[None, :] + j_idx * BLOCK_SIZE_J < J, other=0.0)
            w = tl.load(w_ptrs, mask=offs_j[:, None] + j_idx * BLOCK_SIZE_J < J, other=0.0)

            if dtype_id == 1:
                y = y.to(tl.float16)
                w = w.to(tl.float16)
            elif dtype_id == 2:
                y = y.to(tl.bfloat16)
                w = w.to(tl.bfloat16)
            accumulator = tl.dot(y, w, accumulator, allow_tf32=allow_tf32)
            y_ptrs += BLOCK_SIZE_J * stride_dY_j
            w_ptrs += BLOCK_SIZE_J * stride_wj

        if out_dtype_id == 1:
            dx = accumulator.to(tl.float16)
        elif out_dtype_id == 2:
            dx = accumulator.to(tl.bfloat16)
        else:
            dx = accumulator

        # -----------------------------------------------------------
        # Write the computed block to X (to indices [b, Ind[b,e,k], i]).
        offs_Ind_b = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        valid = offs_Ind_b < B
        Ind_ptr = Ind_ptr + offs_Ind_b * stride_Indb
        offs_xT = tl.load(Ind_ptr, mask=valid, other=0) # list of token ids in the sequence for given batch
        offs_xb = pid_b * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
        offs_xi = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
        x_mask = (offs_xb[:, None] < B) & (offs_xi[None, :] < I)
        offs_xb = offs_xb % B
        offs_xi = offs_xi % I

        x_ptrs = dX_ptr + (offs_xb * stride_dX_b + offs_xT * stride_dX_T)[:, None] + offs_xi[None, :] * stride_dX_i

        tl.atomic_add(x_ptrs, dx, mask=x_mask)


    @triton.autotune(
        configs=get_autotune_config(),
        key=['B', 'J', 'I', 'out_dtype_id', 'allow_tf32', 'dtype_id'], reset_to_zero = ['dW_ptr'],
    )
    @triton.jit
    def expert_gather_backward_dW_kernel(
            # Pointers for a batched tensor:
            X_ptr, dY_ptr, Ind_ptr, dW_ptr,
            # Dimensions:
            B, J, I,
            # Strides for X (shape: [B, T, I])
            stride_X_b, stride_X_t, stride_X_i,
            # Strides for Y (shape: [B, E, K, J])
            stride_dY_b, stride_dY_e, stride_dY_k, stride_dY_j,
            # Stride for Ind (shape: [B, E, K])
            stride_Indb, stride_Inde, stride_Indk,
            # Strides for dW (shape: [E, I, J])
            stride_dW_e, stride_dW_i, stride_dW_j,
            # type parameters
            out_dtype_id: tl.constexpr, allow_tf32: tl.constexpr, dtype_id: tl.constexpr,
            # Meta-parameters:
            BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_J: tl.constexpr, BLOCK_SIZE_I: tl.constexpr,
            # We reuse GROUP_SIZE_B here for tiling over K and N.
            GROUP_SIZE_B: tl.constexpr,
    ):
        """
        Batched backward kernel to compute VJP w.r.t. W for expert_gather.
        \nabla_W[e,i,j] = \sum_{b=1}^{B} \sum_{k=1}^{K} X\bigl[b,\operatorname{Ind}[b,e,k],i\bigr] \cdot \nabla Y[b,e,k,j]
        """
        expert_id = tl.program_id(axis=0)
        pid = tl.program_id(axis=1)
        token_id = tl.program_id(axis=2)

        # Adjust pointers for the current batch.
        dW_ptr = dW_ptr + expert_id * stride_dW_e
        dY_ptr = dY_ptr + expert_id * stride_dY_e + token_id * stride_dY_k
        Ind_ptr = Ind_ptr + expert_id * stride_Inde + token_id * stride_Indk

        # -----------------------------------------------------------
        # The rest is the same tiling strategy as before. (I, J) are the last two dimensions of the output
        num_pid_i = tl.cdiv(I, BLOCK_SIZE_I)
        num_pid_j = tl.cdiv(J, BLOCK_SIZE_J)
        num_pid_in_group = GROUP_SIZE_B * num_pid_j
        group_id = pid // num_pid_in_group
        first_pid_i = group_id * GROUP_SIZE_B
        group_size_i = min(num_pid_i - first_pid_i, GROUP_SIZE_B)
        pid_i = first_pid_i + ((pid % num_pid_in_group) % group_size_i)
        pid_j = (pid % num_pid_in_group) // group_size_i

        offs_xi = (pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)) % I
        offs_yj = (pid_j * BLOCK_SIZE_J + tl.arange(0, BLOCK_SIZE_J)) % J
        xi_mask = offs_xi < I
        yj_mask = offs_yj < J
        offs_b = tl.arange(0, BLOCK_SIZE_B)
        x_ptrs = X_ptr + offs_xi[:, None] * stride_X_i + offs_b[None, :] * stride_X_b
        y_ptrs = dY_ptr + offs_b[:, None] * stride_dY_b + offs_yj[None, :] * stride_dY_j
        Ind_ptrs = Ind_ptr + offs_b * stride_Indb

        accumulator = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_J), dtype=tl.float32)
        for b_idx in range(0, tl.cdiv(B, BLOCK_SIZE_B)):
            mask_valid = offs_b + b_idx * BLOCK_SIZE_B < B

            # Offsets for xT are loaded from the Ind array
            offs_xT = tl.load(Ind_ptrs, mask=mask_valid, other=0.0)

            x_mask = mask_valid[None, :] & xi_mask[:, None]
            x = tl.load(x_ptrs + offs_xT[None, :] * stride_X_t, mask=x_mask, other=0.0)

            y_mask = (offs_b[:, None] + b_idx * BLOCK_SIZE_B < B) & yj_mask[None, :]
            y = tl.load(y_ptrs, mask=y_mask, other=0.0)


            if dtype_id == 1:
                x = x.to(tl.float16)
                y = y.to(tl.float16)
            elif dtype_id == 2:
                x = x.to(tl.bfloat16)
                y = y.to(tl.bfloat16)

            accumulator = tl.dot(x, y, accumulator, allow_tf32=allow_tf32)
            x_ptrs += BLOCK_SIZE_B * stride_X_b
            y_ptrs += BLOCK_SIZE_B * stride_dY_b
            Ind_ptrs += BLOCK_SIZE_B * stride_Indb

        if out_dtype_id == 1:
            dw = accumulator.to(tl.float16)
        elif out_dtype_id == 2:
            dw = accumulator.to(tl.bfloat16)
        else:
            dw = accumulator

        offs_wi = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
        offs_wj = pid_j * BLOCK_SIZE_J + tl.arange(0, BLOCK_SIZE_J)
        w_ptrs = dW_ptr + offs_wi[:, None] * stride_dW_i + offs_wj[None, :] * stride_dW_j
        w_mask = (offs_wi[:, None] < I) & (offs_wj[None, :] < J)

        tl.atomic_add(w_ptrs, dw, mask=w_mask)

    # operations registration for torch.compile

    if version.parse(torch.__version__) >= version.parse("2.2.0"):
        torch.library.define("mylib::expert_gather_forward_call_fn", "(Tensor X, Tensor W, Tensor Ind, ScalarType out_dtype) -> Tensor")
        fw_decorator = torch.library.impl("mylib::expert_gather_forward_call_fn", "default")
    else:
        fw_decorator = lambda x: x

    @fw_decorator
    def expert_gather_forward_call_fn(
            X: torch.Tensor,  W: torch.Tensor, Ind: torch.Tensor, out_dtype=torch.dtype
    ):
        B, _, I = X.shape
        E, _, J = W.shape
        _, _, K = Ind.shape

        Y = torch.empty((B, E, K, J), device=X.device, dtype=X.dtype, requires_grad=X.requires_grad)

        grid = lambda meta: (
            E,
            triton.cdiv(B, meta['BLOCK_SIZE_B']) * triton.cdiv(J, meta['BLOCK_SIZE_J']),
            K,
        )
        expert_gather_forward[grid](
            X, W, Ind, Y,
            B, J, I,
            X.stride(0), X.stride(1), X.stride(2),
            W.stride(0), W.stride(1), W.stride(2),
            Ind.stride(0), Ind.stride(1), Ind.stride(2),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            out_dtype_id=dtype_to_type_id(out_dtype), allow_tf32=False, dtype_id=dtype_to_type_id(out_dtype)
        )

        return Y

    if version.parse(torch.__version__) >= version.parse("2.2.0"):
        torch.library.define("mylib::expert_gather_backward_call_fn", "(Tensor X, Tensor W, Tensor Ind, Tensor grad_output, ScalarType out_dtype) -> (Tensor, Tensor)")
        bw_decorator = torch.library.impl("mylib::expert_gather_backward_call_fn", "default")
    else:
        bw_decorator = lambda x: x

    @bw_decorator
    def expert_gather_backward_call_fn(
            X: torch.Tensor, W: torch.Tensor, Ind: torch.Tensor,
            grad_output: torch.Tensor, dtype: torch.dtype):
        B, _, I = X.shape
        E, _, J = W.shape
        _, _, K = Ind.shape

        grad_X = torch.zeros_like(X)
        grad_W = torch.zeros_like(W)

        grid_dX = lambda meta: (
            E,
            triton.cdiv(B, meta['BLOCK_SIZE_B']) * triton.cdiv(I, meta['BLOCK_SIZE_I']),
            K
        )
        expert_gather_backward_dX_kernel[grid_dX](
            grad_output, W, Ind, grad_X,
            B, J, I,
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
            W.stride(0), W.stride(1), W.stride(2),
            Ind.stride(0), Ind.stride(1), Ind.stride(2),
            grad_X.stride(0), grad_X.stride(1), grad_X.stride(2),
            out_dtype_id=dtype_to_type_id(dtype), allow_tf32=False, dtype_id=dtype_to_type_id(dtype)
        )

        grid_dW = lambda meta: (
            E,
            triton.cdiv(I, meta['BLOCK_SIZE_I']) * triton.cdiv(J, meta['BLOCK_SIZE_J']),
            K
        )
        expert_gather_backward_dW_kernel[grid_dW](
            X, grad_output, Ind, grad_W,
            B,J,I,
            X.stride(0), X.stride(1), X.stride(2),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
            Ind.stride(0), Ind.stride(1), Ind.stride(2),
            grad_W.stride(0), grad_W.stride(1), grad_W.stride(2),
            out_dtype_id=dtype_to_type_id(dtype), allow_tf32=False, dtype_id=dtype_to_type_id(dtype)
        )

        return grad_X, grad_W

    # Function abstract returns a Tensor with the same shape, type and device as the real output
    if version.parse(torch.__version__) >= version.parse("2.2.0"):
        if version.parse(triton.__version__) >= version.parse("3.0.0"):
            abstract_decorator = torch.library.register_fake
        else:
            abstract_decorator = torch.library.impl_abstract

        @abstract_decorator("mylib::expert_gather_forward_call_fn", expert_gather_forward_call_fn)
        def expert_gather_forward_abstract(
            X: torch.Tensor,  W: torch.Tensor, Ind: torch.Tensor, out_dtype=torch.dtype
        ):
            B, _, I = X.shape
            E, _, J = W.shape
            _, _, K = Ind.shape

            Y = torch.empty((B, E, K, J), device=X.device, dtype=X.dtype, requires_grad=X.requires_grad)
            return Y

        @abstract_decorator("mylib::expert_gather_backward_call_fn", expert_gather_backward_call_fn)
        def expert_gather_backward_abstract(
            X: torch.Tensor,  W: torch.Tensor, Ind: torch.Tensor, grad_output: torch.Tensor, out_dtype=torch.dtype
        ):
            return X, W
        
    if version.parse(torch.__version__) >= version.parse("2.2.0"):
        expert_gather_forward_call = torch.ops.mylib.expert_gather_forward_call_fn
        expert_gather_backward_call = torch.ops.mylib.expert_gather_backward_call_fn
    else:
        expert_gather_forward_call = expert_gather_forward_call_fn
        expert_gather_backward_call = expert_gather_backward_call_fn
        


class ExpertGatherFunction(torch.autograd.Function):
    """
    Torch Interface to the expert_gather kernels
    """
    @staticmethod
    def forward(ctx, X, W, Ind, out_type=torch.float32):
        """
        Forward pass computes expert_gather
        Y[b,e,k,j] = \sum_{i=1}^{I} X\bigl[b,\operatorname{Ind}[b,e,k],i\bigr] \cdot W[e,i,j]
        Inputs:
            X: Tensor of shape [B, T, I]
            W: Tensor of shape [E, I, J]
            Ind: Tensor of shape [B, E, K]
            out_type: type to use in the calculations (torch.float32 recommended)
            allow_tf32: whether to use tf32 for the accumulation of the tensors in the kernel (faster but lower precision)
        Returns:
            Y: Tensor of shape [B, E, K, J]
        """
        ctx.save_for_backward(X,W, Ind)
        ctx.dtype = out_type
        return expert_gather_forward_call(X, W, Ind, out_type)

    @staticmethod
    def backward(ctx, grad_output):
        X, W, Ind = ctx.saved_tensors

        grad_X, grad_W = expert_gather_backward_call(X, W, Ind, grad_output, ctx.dtype)
        return grad_X, grad_W, None, None


def expert_gather_function(X: torch.Tensor, W: torch.Tensor, Ind: torch.Tensor, out_dtype=torch.float32):
     # Torch 2.2 on Volta GPUs is broken.
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

    return ExpertGatherFunction.apply(X, W, Ind, out_dtype)

class ExpertGather(nn.Module):
    def __init__(self, E, I, J, bias=False):
        super().__init__()
        self.E, self.I, self.J = E, I, J
        self.W = nn.Parameter(torch.Tensor(E, I, J))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(E, J))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


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

        y = expert_gather_function(x, self.W, Ind)
        if self.bias:
            y = y + self.bias # todo: fix, now shapes don't match, output should have shape [B, K, E, J]
        return y

    def to(self, device, *args, **kwargs):
        self.W = nn.Parameter(self.W.to(device, *args, **kwargs))
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.to(device, *args, **kwargs))
        return self