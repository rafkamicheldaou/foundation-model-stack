from typing import Optional

import torch
import torch.nn as nn

from fms.utils.activation import str_to_activation


# All the helper functions are identical to ssm.py (Default SSM Module)
def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int):
    """
    Padding x tensor with `pad_size` on the seq_len dim (dim=1)

    Assumes that we only have tensors of either size 4 or 3
    """
    pad_shape = (
        (0, 0, 0, 0, 0, pad_size, 0, 0)
        if len(input_tensor.shape) == 4
        else (0, 0, 0, pad_size, 0, 0)
    )

    return torch.nn.functional.pad(input_tensor, pad_shape, mode="constant", value=0)


def reshape_into_chunks(input_tensor, pad_size, chunk_size):
    """
    Padding input_tensor with `pad_size` on the seq_len dim (dim=1) and
    simultaneously splitting it into chunk sequences.

    Assumes that we only have tensors of either size 4 or 3
    """
    # [bsz, seq_len, ...] -> [bsz, seq_len multiple of chunk_size, ...]
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)

    if len(input_tensor.shape) == 3:
        # [bsz, seq_len multiple of chunk_size, nheads] -> [bsz, -1, chunk_size, nheads]
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2]
        )
    else:
        # [bsz, seq_len multiple of chunk_size, nheads, head_dim or state_size] -> [bsz, -1, chunk_size, nheads, head_dim or state_size]
        return input_tensor.reshape(
            input_tensor.shape[0],
            -1,
            chunk_size,
            input_tensor.shape[2],
            input_tensor.shape[3],
        )


def segment_sum(input_tensor):
    """
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    chunk_size = input_tensor.size(-1)
    # 1. expand input tensor to have an additional dimension and repeat along that dimension
    # [..., chunk_size] -> [..., chunk_size, chunk_size]
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    # 2. create a lower triangular mask with the diagonal set to 0 to 0 out elements above diag
    mask = torch.tril(
        torch.ones(
            chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool
        ),
        diagonal=-1,
    )
    input_tensor = input_tensor.masked_fill(~mask, 0)
    # 3. compute actual cumsum
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)

    # 4. apply mask to keep only the lower triangular part of the cumulative sum result (incl diagonal this time)
    mask = torch.tril(
        torch.ones(
            chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool
        ),
        diagonal=0,
    )
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
            hidden_states = hidden_states * nn.functional.silu(gate.to(torch.float32))
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states.to(input_dtype)

# SSMCacheUnit is not used by this Independent SSM module, kept for consistency
class SSMCacheUnit:
    def __init__(
        self,
        emb_dim: int,
        nheads: int,
        head_dim: int,
        conv_kernel,
        expand: float,
        n_groups: int,
        state_size: int,
        batch_size: int,
        dtype: torch.dtype,
        device: Optional[str] = None,
    ):
        self.seqlen_offset = 0
        self.dtype = dtype
        self.conv_kernel_size = conv_kernel
        self.intermediate_size = int(expand * emb_dim)
        self.has_previous_state = False

        self.conv_state = torch.zeros(
            batch_size,
            self.intermediate_size + 2 * n_groups * state_size,
            self.conv_kernel_size,
            device=device,
            dtype=dtype,
        )
        self.ssm_state = torch.zeros(
            batch_size, nheads, head_dim, state_size, device=device, dtype=dtype
        )

    def update_conv_state(
        self, new_conv_state: torch.Tensor, cache_position: torch.Tensor
    ):
        conv_state = self.conv_state
        cache_position = cache_position.clamp(0, self.conv_kernel_size - 1)

        conv_state = conv_state.roll(shifts=-1, dims=-1)
        conv_state[:, :, cache_position] = new_conv_state.to(conv_state.device)
        self.conv_state.zero_()
        self.conv_state += conv_state
        return self.conv_state


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    if (
        attention_mask is not None
        and attention_mask.shape[1] > 1
        and attention_mask.shape[0] > 1
    ):
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * (attention_mask[:, -1, :, None] == 0)).to(
            dtype
        )

    return hidden_states


# Independent SSM module: identical to ssm.py (the Default SSM module), 
# except no inter-chunk state transfer and no recurrence across chunks.
# Removed any parts of the Default module relating to the past state. 
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
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size = conv_kernel,
            padding = conv_kernel - 1,
            groups = self.conv_dim,
            bias = use_conv_bias,
        )

        proj_size = self.intermediate_size + self.conv_dim + nheads
        self.in_proj = nn.Linear(emb_dim, proj_size, bias=use_bias)

        self.dt_bias = nn.Parameter(torch.ones(nheads))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, nheads + 1)))
        self.D = nn.Parameter(torch.ones(nheads))

        self.norm = RMSNormGated(self.intermediate_size, eps=norm_eps)
        self.out_proj = nn.Linear(self.intermediate_size, emb_dim, bias=use_bias)

        self.time_step_limit = (0.0, float("inf"))

    def forward(
        self,
        input_states,
        mask,
        past_key_value_state: Optional[SSMCacheUnit] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch_size, seq_len, _ = input_states.shape

        # 1. Gated MLP projection
        input_states = apply_mask_to_padding_states(input_states, mask)
        proj = self.in_proj(input_states)
        gate, h_BC, dt = proj.split(
            [self.intermediate_size, self.conv_dim, self.nheads], dim=-1
        )

        # 2. Convolution sequence transformation
        h_BC = self.act(
            self.conv1d(h_BC.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        )

        # Split into hidden / B / C
        h_BC = apply_mask_to_padding_states(h_BC, mask)
        hidden, B, C = torch.split(
            h_BC,
            [self.intermediate_size,
             self.n_groups * self.ssm_state_size,
             self.n_groups * self.ssm_state_size],
            dim=-1,
        )

        # 3. SSM transformation without inter-chunk recurrence
        # Closely following the no use_precomputed_states branch of Default SSM
        A = -torch.exp(self.A_log.float())  # [num_heads]
        dt = F.softplus(dt + self.dt_bias).clamp(*self.time_step_limit)

        H = hidden.reshape(batch_size, seq_len, -1, self.head_dim).float()
        B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
        C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
        B = B.repeat(1, 1, self.nheads // self.n_groups, 1)
        C = C.repeat(1, 1, self.nheads // self.n_groups, 1)

        pad = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
        D_resid = self.D[:, None] * pad_tensor_by_size(H, pad)

        H = H * dt[..., None]
        A = A.to(H.dtype) * dt

        # Rearrange into blocks/chunks
        H, A, B, C = [
            reshape_into_chunks(t, pad, self.chunk_size)
            for t in (H, A, B, C)
        ]

        # [bsz, -1, chunk_size, num_heads] -> [bsz, num_heads, -1, chunk_size]
        A_perm = A.permute(0, 3, 1, 2)
        A_cum = torch.cumsum(A_perm, dim=-1)
        # Compute the output for each intra-chunk (diagonal blocks)
        # This is the analog of a causal mask
        L_tri = torch.exp(segment_sum(A_perm))

        # Contraction of C and B to get G (attention-weights like)
        G  = (C[:, :, :, None] * B[:, :, None]).sum(-1)  # shape: (b, c, l, s, h)

        # Compute M, equivalent to applying attention mask to weights
        M  = G * L_tri.permute(0, 2, 3, 4, 1)

        # Compute Y_diag (apply to values)
        Yd = (M[..., None] * H[:, :, None]).sum(3)

        # Compute the state for each intra-chunk
        # (right term of low-rank factorization of off-diagonal blocks; B terms)
        decay = torch.exp(A_cum[..., -1:] - A_cum)
        B_dec = B * decay.permute(0, 2, 3, 1)[..., None]
        state = (B_dec[..., None] * H[..., None]).sum(2)

        # Skipping the inter-chunk SSM recurrence part here (state not updated further)

        # Compute state -> output conversion per chunk
        # (left term of low-rank factorization of off-diagonal blocks; C terms)
        Sd_out = torch.exp(A_cum)
        CofS = C[..., None] * state[:, :, None]
        Yoff = CofS.sum(-1) * Sd_out.permute(0, 2, 3, 1)[..., None]

        # Add output of diagonal and off-diagonal blocks
        y = Yd + Yoff
        # [bsz, -1, self.chunk_size, num_heads, head_dim] -> [bsz, (padded) seq_len, num_heads, head_dim]
        y = y.reshape(batch_size, -1, self.nheads, self.head_dim) + D_resid
        # Cutting off padded chunks
        if pad:
            y = y[:, :seq_len]
        y = y.reshape(batch_size, seq_len, -1)

        # Skipping the init cache part here (we are not storing any hidden states)

        # 4. RMSNorm and final linear projection
        # [batch, seq_len, hidden_size] for the output
        # None is replacing past_key_value_state from the Default module 
        # to stay consistent with the overarching Bamba architecture
        return self.out_proj(self.norm(y, gate)), None
