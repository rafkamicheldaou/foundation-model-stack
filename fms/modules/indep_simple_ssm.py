from typing import Optional

import torch
import torch.nn as nn

from fms.utils.activation import str_to_activation


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
        super(SSM, self).__init__()
        self.nheads = nheads
        self.emb_dim = emb_dim
        self.ssm_state_size = state_size
        self.conv_kernel_size = conv_kernel
        self.intermediate_size = int(expand * self.emb_dim)
        self.use_conv_bias = use_conv_bias
        self.activation = activation_fn
        self.act = str_to_activation(activation_fn)

        self.layer_norm_epsilon = norm_eps
        self.n_groups = n_groups
        self.head_dim = head_dim
        self.chunk_size = chunk_size

        # convolution + gated-MLP dims
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # projection of the input hidden states
        projection_size = self.intermediate_size + self.conv_dim + self.nheads
        self.in_proj = nn.Linear(
            self.emb_dim,
            projection_size,
            bias=use_bias,
        )

        # selective projection used to make dt, B and C input dependant
        self.dt_bias = nn.Parameter(torch.ones(self.nheads))

        # S4D real initialization. These are not discretized!
        A = torch.arange(1, self.nheads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.nheads))

        # gated RMSNorm
        self.norm = RMSNormGated(self.intermediate_size, eps=self.layer_norm_epsilon)

        self.time_step_limit = (0.0, float("inf"))
        self.out_proj = nn.Linear(self.intermediate_size, self.emb_dim, bias=use_bias)

    def forward(
        self,
        input_states: torch.Tensor,
        mask: torch.Tensor,
        past_key_value_state: Optional[SSMCacheUnit] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        # 1. Gated MLP's linear projection
        input_states = apply_mask_to_padding_states(input_states, mask)
        projected_states = self.in_proj(input_states)
        gate, hidden_states_B_C, dt = projected_states.split(
            [self.intermediate_size, self.conv_dim, self.nheads], dim=-1
        )

        use_precomputed_states = (
            past_key_value_state is not None
            and past_key_value_state.has_previous_state
            and seq_len == 1
            and cache_position is not None
        )

        # 2. Convolution sequence transformation
        if use_precomputed_states:
            # Single-step (auto-regressive) path
            past_key_value_state.conv_state = past_key_value_state.conv_state.roll(
                shifts=-1, dims=-1
            )
            past_key_value_state.conv_state[:, :, -1] = hidden_states_B_C[:, 0, :].to(
                past_key_value_state.conv_state.device
            )
            conv_states = past_key_value_state.conv_state.to(self.conv1d.weight.device)
            hidden_states_B_C = torch.sum(
                conv_states * self.conv1d.weight.squeeze(1), dim=-1
            )
            if self.use_conv_bias:
                hidden_states_B_C = hidden_states_B_C + self.conv1d.bias
            hidden_states_B_C = self.act(hidden_states_B_C)
        else:
            # Full-sequence path
            hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)

            # CHANGED from Default: drop storing conv_state so no chunk info flows
            hidden_states_B_C = self.act(
                self.conv1d(hidden_states_B_C_transposed)[..., :seq_len].transpose(1, 2)
            )

        hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, mask)

        # Split into hidden_states, B, C
        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [
                self.intermediate_size,
                self.n_groups * self.ssm_state_size,
                self.n_groups * self.ssm_state_size,
            ],
            dim=-1,
        )

        # 3. SSM transformation
        A = -torch.exp(self.A_log.float())

        if use_precomputed_states:
            # Single-step (auto-regressive) path unchanged
            dt_step = dt[:, 0, :][:, None, ...].transpose(1, 2).expand(
                batch_size, self.nheads, self.head_dim
            )
            dt_step = torch.nn.functional.softplus(dt_step + self.dt_bias[..., None])
            dt_step = torch.clamp(dt_step, *self.time_step_limit)

            A_mat = A[..., None, None].expand(self.nheads, self.head_dim, self.ssm_state_size)
            dA = torch.exp(dt_step[..., None] * A_mat).to(
                past_key_value_state.ssm_state.device
            )

            B_proj = (
                B.reshape(batch_size, self.n_groups, -1)[..., None, :]
                .expand(
                    batch_size,
                    self.n_groups,
                    self.nheads // self.n_groups,
                    B.shape[-1],
                )
                .contiguous()
                .reshape(batch_size, -1, B.shape[-1])
            )
            dB = dt_step[..., None] * B_proj[..., None, :]

            hidden_reshaped = hidden_states.reshape(batch_size, -1, self.head_dim)
            dBx = (dB * hidden_reshaped[..., None]).to(
                past_key_value_state.ssm_state.device
            )
            past_key_value_state.ssm_state.copy_(
                past_key_value_state.ssm_state * dA + dBx
            )

            C_proj = (
                C.reshape(batch_size, self.n_groups, -1)[..., None, :]
                .expand(
                    batch_size,
                    self.n_groups,
                    self.nheads // self.n_groups,
                    C.shape[-1],
                )
                .contiguous()
                .reshape(batch_size, -1, C.shape[-1])
            )
            ssm_states = past_key_value_state.ssm_state.to(
                device=C_proj.device, dtype=C_proj.dtype
            )
            s_flat = ssm_states.view(batch_size * self.nheads, self.head_dim, self.ssm_state_size)
            C_flat = C_proj.view(batch_size * self.nheads, self.ssm_state_size, 1)
            y = torch.bmm(s_flat, C_flat).view(batch_size, self.nheads, self.head_dim)

            D_mat = self.D[..., None].expand(self.nheads, self.head_dim)
            y = (y + hidden_reshaped * D_mat).to(y.dtype)
            y = y.view(batch_size, -1)[:, None, ...]
        else:
            # CHANGED from Default: drop cross-chunk recurrence for fully independent chunks
            dt = torch.nn.functional.softplus(dt + self.dt_bias)
            dt = torch.clamp(dt, *self.time_step_limit)

            hid = hidden_states.reshape(batch_size, seq_len, self.nheads, self.head_dim)
            pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

            D_residual = self.D[..., None] * pad_tensor_by_size(hid, pad_size)

            hid = hid * dt[..., None]
            A_scaled = A.to(hid.dtype) * dt

            # chunk each
            hid_chunks, A_chunks, B_chunks, C_chunks = [
                reshape_into_chunks(t, pad_size, self.chunk_size)
                for t in (hid, A_scaled, B.reshape(batch_size, seq_len, -1, self.ssm_state_size), C.reshape(batch_size, seq_len, -1, self.ssm_state_size))
            ]
            B_chunks = B_chunks.repeat(1, 1, self.nheads // self.n_groups, 1)
            C_chunks = C_chunks.repeat(1, 1, self.nheads // self.n_groups, 1)

            A_chunks = A_chunks.permute(0, 3, 1, 2)
            A_cumsum = torch.cumsum(A_chunks, dim=-1)

            # 1. Compute the output for each intra-chunk (diagonal blocks)
            L = torch.exp(segment_sum(A_chunks))
            G_int = C_chunks[..., None, :, :] * B_chunks[:, :, None, :, :]
            G = G_int.sum(dim=-1)
            M_int = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
            M = M_int.sum(dim=-1)
            Y_diag = (M[..., None] * hid_chunks[:, :, None]).sum(dim=3)

            # 2. Compute the state for each intra-chunk
            decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
            B_decay = B_chunks * decay_states.permute(0, -2, -1, 1)[..., None]
            states = (B_decay[..., None, :] * hid_chunks[..., None]).sum(dim=2)

            # CHANGED from Default: skip inter-chunk recurrence

            # 3. Compute state -> output conversion within each chunk
            state_decay_out = torch.exp(A_cumsum)
            C_times_states = C_chunks[..., None, :] * states[:, :, None, ...]
            state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
            Y_off = C_times_states.sum(-1) * state_decay_out_permuted[..., None]

            y = Y_diag + Y_off
            y = y.view(batch_size, -1, self.nheads, self.head_dim)
            y = y + D_residual
            if pad_size > 0:
                y = y[:, :seq_len, :, :]
            y = y.reshape(batch_size, seq_len, -1)

        # 4. Final linear projection
        scan_output = self.norm(y, gate)
        contextualized_states = self.out_proj(scan_output.to(dtype))
        return contextualized_states, past_key_value_state
