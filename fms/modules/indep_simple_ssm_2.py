import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from fms.utils.activation import str_to_activation

# Helper functions

def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int):
    pad_shape = (
        (0, 0, 0, 0, 0, pad_size, 0, 0)
        if len(input_tensor.shape) == 4
        else (0, 0, 0, pad_size, 0, 0)
    )
    return F.pad(input_tensor, pad_shape, mode="constant", value=0)

def reshape_into_chunks(input_tensor, pad_size, chunk_size):
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)
    if len(input_tensor.shape) == 3:
        return input_tensor.reshape(input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2])
    else:
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2], input_tensor.shape[3]
        )

def segment_sum(input_tensor):
    chunk_size = input_tensor.size(-1)
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum

class RMSNormGated(nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(emb_dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class SSM(nn.Module):
    def __init__(
        self,
        nheads: int,
        emb_dim: int,
        state_size: int,
        conv_kernel: int,
        expand: float,
        use_bias: bool,
        use_conv_bias: bool,
        activation_fn: str,
        norm_eps: float,
        n_groups: int,
        head_dim: int,
        chunk_size: int,
    ):
        super().__init__()
        self.nheads = nheads
        self.emb_dim = emb_dim
        self.ssm_state_size = state_size
        self.conv_kernel_size = conv_kernel
        self.intermediate_size = int(expand * emb_dim)
        self.n_groups = n_groups
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.act = str_to_activation(activation_fn)

        self.conv_dim = self.intermediate_size + 2 * n_groups * state_size
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, kernel_size=conv_kernel, padding=conv_kernel - 1, groups=self.conv_dim, bias=use_conv_bias)
        proj_size = self.intermediate_size + self.conv_dim + nheads
        self.in_proj = nn.Linear(emb_dim, proj_size, bias=use_bias)
        self.dt_bias = nn.Parameter(torch.ones(nheads))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, nheads + 1)))
        self.D = nn.Parameter(torch.ones(nheads))
        self.norm = RMSNormGated(self.intermediate_size, eps=norm_eps)
        self.out_proj = nn.Linear(self.intermediate_size, emb_dim, bias=use_bias)
        self.time_step_limit = (0.0, float("inf"))

    def forward(self, input_states, mask, **kwargs):
        batch_size, seq_len, _ = input_states.shape
        input_states = input_states.clone()
        proj = self.in_proj(input_states)
        gate, h_BC, dt = proj.split([self.intermediate_size, self.conv_dim, self.nheads], dim=-1)
        h_BC = self.act(self.conv1d(h_BC.transpose(1, 2))[..., :seq_len].transpose(1, 2))
        hidden, B, C = torch.split(h_BC, [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size], dim=-1)

        A = -torch.exp(self.A_log.float())
        dt = F.softplus(dt + self.dt_bias).clamp(*self.time_step_limit)

        H = hidden.reshape(batch_size, seq_len, -1, self.head_dim).float()
        B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
        C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
        B = B.repeat(1, 1, self.nheads // self.n_groups, 1)
        C = C.repeat(1, 1, self.nheads // self.n_groups, 1)

        pad = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
        H_padded = pad_tensor_by_size(H, pad)
        D_resid = self.D[:, None] * H_padded

        H_scaled = H * dt[..., None]
        A_scaled = A.to(H.dtype) * dt

        H_chunks, A_chunks, B_chunks, C_chunks = [reshape_into_chunks(t, pad, self.chunk_size) for t in (H_scaled, A_scaled, B, C)]

        A_perm = A_chunks.permute(0, 3, 1, 2)
        A_cum = torch.cumsum(A_perm, dim=-1)
        L_tri = torch.exp(segment_sum(A_perm))

        G = (C_chunks[:, :, :, None] * B_chunks[:, :, None]).sum(-1)
        M = G * L_tri.permute(0, 2, 3, 4, 1)
        Yd = (M[..., None] * H_chunks[:, :, None]).sum(3)

        decay = torch.exp(A_cum[..., -1:] - A_cum)
        B_dec = B_chunks * decay.permute(0, 2, 3, 1)[..., None]

        b_dec_shape = B_dec.shape
        h_chunks_shape = H_chunks.shape
        mismatch_dim = -2
        if b_dec_shape[mismatch_dim] != h_chunks_shape[mismatch_dim]:
            smaller_tensor = B_dec if b_dec_shape[mismatch_dim] < h_chunks_shape[mismatch_dim] else H_chunks
            larger_shape = h_chunks_shape if smaller_tensor is B_dec else b_dec_shape
            pad_size = [0] * (2 * len(larger_shape))
            diff = larger_shape[mismatch_dim] - smaller_tensor.shape[mismatch_dim]
            pad_size[2 * (len(larger_shape) + mismatch_dim) + 1] = diff
            smaller_tensor = F.pad(smaller_tensor, pad_size, mode="constant", value=0)
            if smaller_tensor is B_dec:
                B_dec = smaller_tensor
            else:
                H_chunks = smaller_tensor

        state = (B_dec[..., None] * H_chunks[..., None]).sum(2)

        Sd_out = torch.exp(A_cum)
        CofS = C_chunks[..., None] * state[:, :, None]
        Yoff = CofS.sum(-1) * Sd_out.permute(0, 2, 3, 1)[..., None]

        y = Yd + Yoff
        y = y.reshape(batch_size, -1, self.nheads, self.head_dim) + D_resid
        if pad:
            y = y[:, :seq_len]
        y = y.reshape(batch_size, seq_len, -1)

        y = self.norm(y, gate)
        return self.out_proj(y), None
