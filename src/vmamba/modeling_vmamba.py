# Copyright 2024 MzeroMiko, Derk Mus, state-spaces/mamba, and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch VMamba model.

A Hugging Face Transformers port of VMamba (Visual State Space Model). The
architecture matches the official v2 checkpoints (`v05_noz`, channel-first,
patch-embed v2, downsample v3). Selective scan is implemented in pure PyTorch
(parallel prefix scan), so `mamba_ssm`, custom CUDA kernels, triton, timm, and
yacs are not required.

References:
- Liu et al., "VMamba: Visual State Space Model" (https://arxiv.org/abs/2401.10166)
- Hugging Face `models/mamba` selective-scan style
- https://github.com/MzeroMiko/VMamba
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from transformers.activations import ACT2FN
from transformers.backbone_utils import BackboneMixin, filter_output_hidden_states
from transformers.modeling_outputs import (
    BackboneOutput,
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, logging
from transformers.utils.generic import can_return_tuple

from .configuration_vmamba import VMambaConfig


logger = logging.get_logger(__name__)

VMAMBA_PRETRAINED_MODEL_ARCHIVE_LIST = []

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as mamba_selective_scan_fn
except Exception:
    mamba_selective_scan_fn = None


# ---------------------------------------------------------------------------
# Selective scan (HF Mamba-style: pure PyTorch, optional mamba_ssm)
# ---------------------------------------------------------------------------


def _parallel_prefix_scan(delta_a: torch.Tensor, delta_b_u: torch.Tensor) -> torch.Tensor:
    """Inclusive prefix scan of the affine recurrence ``h_t = a_t * h_{t-1} + b_t``.

    Args:
        delta_a: ``(batch, dim, length, state)``
        delta_b_u: ``(batch, dim, length, state)``

    Returns:
        Hidden states ``h`` of shape ``(batch, dim, length, state)``.
    """
    a = delta_a
    b = delta_b_u
    length = a.shape[2]
    step = 1
    while step < length:
        a_prev = a
        b_prev = b
        a = a.clone()
        b = b.clone()
        b[:, :, step:] = a_prev[:, :, step:] * b_prev[:, :, :-step] + b_prev[:, :, step:]
        a[:, :, step:] = a_prev[:, :, step:] * a_prev[:, :, :-step]
        step *= 2
    return b


def _sequential_scan(delta_a: torch.Tensor, delta_b_u: torch.Tensor) -> torch.Tensor:
    """Reference sequential scan. Used in tests and as a tiny-sequence fallback."""
    batch, dim, length, state = delta_a.shape
    h = delta_a.new_zeros(batch, dim, state)
    outputs = []
    for t in range(length):
        h = delta_a[:, :, t] * h + delta_b_u[:, :, t]
        outputs.append(h)
    return torch.stack(outputs, dim=2)


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    a_matrix: torch.Tensor,
    b_matrix: torch.Tensor,
    c_matrix: torch.Tensor,
    d_vector: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = True,
    use_associative_scan: bool = True,
) -> torch.Tensor:
    """Selective scan used by SS2D.

    Args:
        u: ``(batch, groups * channels, length)``
        delta: ``(batch, groups * channels, length)``
        a_matrix: ``(groups * channels, state)``
        b_matrix: ``(batch, groups, state, length)``
        c_matrix: ``(batch, groups, state, length)``
        d_vector: ``(groups * channels,)`` skip parameter
        delta_bias: ``(groups * channels,)``
    """
    if mamba_selective_scan_fn is not None and u.is_cuda:
        return mamba_selective_scan_fn(
            u,
            delta,
            a_matrix,
            b_matrix,
            c_matrix,
            d_vector,
            None,
            delta_bias,
            delta_softplus,
        )

    dtype_in = u.dtype
    batch, groups, state, length = b_matrix.shape
    dim = u.shape[1]
    channels = dim // groups

    if delta_bias is not None:
        delta = delta + delta_bias[..., None].to(dtype=delta.dtype)
    if delta_softplus:
        delta = F.softplus(delta)

    u_f = u.float()
    delta_f = delta.float()
    a_f = a_matrix.float()
    b_f = b_matrix.float().unsqueeze(2).expand(batch, groups, channels, state, length)
    b_f = b_f.reshape(batch, dim, state, length)
    c_f = c_matrix.float().unsqueeze(2).expand(batch, groups, channels, state, length)
    c_f = c_f.reshape(batch, dim, state, length)

    # deltaA: (B, D, L, N);  A is (D, N)
    delta_a = torch.exp(delta_f.unsqueeze(-1) * a_f.unsqueeze(0).unsqueeze(2))
    delta_b_u = delta_f.unsqueeze(-1) * b_f.transpose(2, 3) * u_f.unsqueeze(-1)

    if use_associative_scan and length > 1:
        states = _parallel_prefix_scan(delta_a, delta_b_u)
    else:
        states = _sequential_scan(delta_a, delta_b_u)

    y = (states * c_f.transpose(2, 3)).sum(-1)  # (B, D, L)
    if d_vector is not None:
        y = y + u_f * d_vector.float().unsqueeze(0).unsqueeze(-1)
    return y.to(dtype=dtype_in)


# ---------------------------------------------------------------------------
# 2D cross-scan / cross-merge (official scans=0 / "cross2d")
# ---------------------------------------------------------------------------


def cross_scan(hidden_states: torch.Tensor) -> torch.Tensor:
    """Traverse a feature map along four routes.

    Args:
        hidden_states: ``(batch, channels, height, width)``

    Returns:
        ``(batch, 4, channels, height * width)``
    """
    batch, channels, height, width = hidden_states.shape
    length = height * width
    scanned = hidden_states.new_empty(batch, 4, channels, length)
    scanned[:, 0] = hidden_states.flatten(2, 3)
    scanned[:, 1] = hidden_states.transpose(2, 3).flatten(2, 3)
    scanned[:, 2:4] = torch.flip(scanned[:, 0:2], dims=[-1])
    return scanned


def cross_merge(scanned: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Inverse of [`cross_scan`]: merge four scanned sequences back to a map.

    Args:
        scanned: ``(batch, 4, channels, height, width)`` or ``(..., height * width)``
        height, width: spatial size of the original map.

    Returns:
        ``(batch, channels, height * width)``
    """
    batch, _, channels, *_ = scanned.shape
    scanned = scanned.reshape(batch, 4, channels, -1)
    merged = scanned[:, 0:2] + scanned[:, 2:4].flip(dims=[-1])
    row = merged[:, 0]
    col = merged[:, 1].reshape(batch, channels, width, height).transpose(2, 3).reshape(batch, channels, -1)
    return row + col


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class VMambaLinear2d(nn.Linear):
    """`nn.Linear` stored as `(out, in)` but applied as a 1x1 conv on NCHW maps.

    Matches the official `Linear2d`, so original checkpoints load without a
    reshape.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.conv2d(hidden_states, self.weight[:, :, None, None], self.bias)


class VMambaLayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channel axis of an NCHW feature map."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.permute(0, 2, 3, 1)
        hidden_states = F.layer_norm(hidden_states, self.normalized_shape, self.weight, self.bias, self.eps)
        return hidden_states.permute(0, 3, 1, 2)


class VMambaDropPath(nn.Module):
    """Stochastic depth (DropPath) per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return hidden_states
        keep_prob = 1.0 - self.drop_prob
        shape = (hidden_states.shape[0],) + (1,) * (hidden_states.ndim - 1)
        random_tensor = torch.rand(shape, dtype=hidden_states.dtype, device=hidden_states.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return hidden_states.div(keep_prob) * random_tensor

    def extra_repr(self) -> str:
        return f"p={self.drop_prob}"


def _init_dt_bias(d_inner: int, dt_min: float = 0.001, dt_max: float = 0.1, dt_init_floor: float = 1e-4) -> torch.Tensor:
    dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
    # Inverse softplus: https://github.com/pytorch/pytorch/issues/72759
    return dt + torch.log(-torch.expm1(-dt))


class VMambaSS2D(nn.Module):
    """2D Selective Scan (SS2D) — the core of a VSS block.

    Official v2 default is `v05_noz`: no SiLU gate, `conv1d` projections, four-way
    cross-scan. This implementation is backend-agnostic (no custom CUDA).
    """

    def __init__(self, config: VMambaConfig, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.d_state = config.ssm_state_size
        self.d_inner = int(config.ssm_ratio * dim)
        self.dt_rank = int(math.ceil(dim / 16) if config.ssm_dt_rank == "auto" else config.ssm_dt_rank)
        self.k_group = 4
        self.channel_first = config.channel_first
        self.use_gate = config.ssm_use_gate
        self.use_associative_scan = config.use_associative_scan
        self.with_dconv = config.ssm_conv_kernel > 1

        linear_cls = VMambaLinear2d if config.channel_first else nn.Linear
        d_proj = self.d_inner * (2 if self.use_gate else 1)
        self.in_proj = linear_cls(dim, d_proj, bias=config.use_bias)
        self.act = ACT2FN[config.hidden_act]

        if self.with_dconv:
            self.conv2d = nn.Conv2d(
                self.d_inner,
                self.d_inner,
                kernel_size=config.ssm_conv_kernel,
                padding=(config.ssm_conv_kernel - 1) // 2,
                groups=self.d_inner,
                bias=config.ssm_conv_bias,
            )
        else:
            self.conv2d = None

        x_proj = [
            nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False) for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([proj.weight for proj in x_proj], dim=0))
        del x_proj

        self.out_norm = (
            VMambaLayerNorm2d(self.d_inner, eps=config.layer_norm_eps)
            if config.channel_first
            else nn.LayerNorm(self.d_inner, eps=config.layer_norm_eps)
        )
        self.out_proj = linear_cls(self.d_inner, dim, bias=config.use_bias)
        self.dropout = nn.Dropout(config.hidden_dropout_prob) if config.hidden_dropout_prob > 0.0 else nn.Identity()

        self._init_ssm_parameters(config)

    def _init_ssm_parameters(self, config: VMambaConfig) -> None:
        k_group, d_inner, d_state, dt_rank = self.k_group, self.d_inner, self.d_state, self.dt_rank
        if config.ssm_init == "v0":
            dt_weights = []
            dt_biases = []
            for _ in range(k_group):
                dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
                dt_init_std = dt_rank**-0.5
                nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
                with torch.no_grad():
                    dt_proj.bias.copy_(_init_dt_bias(d_inner))
                dt_weights.append(dt_proj.weight)
                dt_biases.append(dt_proj.bias)
            self.dt_projs_weight = nn.Parameter(torch.stack(dt_weights, dim=0))
            self.dt_projs_bias = nn.Parameter(torch.stack(dt_biases, dim=0))
            a = torch.arange(1, d_state + 1, dtype=torch.float32).view(1, -1).repeat(d_inner, 1)
            a_log = torch.log(a)[None].repeat(k_group, 1, 1).flatten(0, 1)
            self.A_logs = nn.Parameter(a_log)
            self.Ds = nn.Parameter(torch.ones(k_group * d_inner))
        elif config.ssm_init == "v1":
            self.dt_projs_weight = nn.Parameter(0.1 * torch.randn(k_group, d_inner, dt_rank))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.randn(k_group, d_inner))
            self.A_logs = nn.Parameter(torch.randn(k_group * d_inner, d_state))
            self.Ds = nn.Parameter(torch.ones(k_group * d_inner))
        elif config.ssm_init == "v2":
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand(k_group, d_inner, dt_rank))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand(k_group, d_inner))
            self.A_logs = nn.Parameter(torch.zeros(k_group * d_inner, d_state))
            self.Ds = nn.Parameter(torch.ones(k_group * d_inner))
        else:
            raise ValueError(f"Unknown ssm_init={config.ssm_init}")
        self.A_logs._no_weight_decay = True
        self.Ds._no_weight_decay = True

    def _forward_core(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, dim, height, width = hidden_states.shape
        length = height * width
        groups, state, rank = self.k_group, self.d_state, self.dt_rank

        xs = cross_scan(hidden_states)  # (B, K, C, L)
        x_dbl = F.conv1d(
            xs.reshape(batch, -1, length),
            self.x_proj_weight.reshape(-1, dim, 1),
            groups=groups,
        ).reshape(batch, groups, -1, length)
        dts, bs, cs = torch.split(x_dbl, [rank, state, state], dim=2)
        dts = F.conv1d(
            dts.reshape(batch, -1, length),
            self.dt_projs_weight.reshape(groups * dim, -1, 1),
            groups=groups,
        )

        ys = selective_scan(
            u=xs.reshape(batch, -1, length),
            delta=dts,
            a_matrix=-self.A_logs.float().exp(),
            b_matrix=bs.contiguous(),
            c_matrix=cs.contiguous(),
            d_vector=self.Ds.float(),
            delta_bias=self.dt_projs_bias.reshape(-1).float(),
            delta_softplus=True,
            use_associative_scan=self.use_associative_scan,
        ).reshape(batch, groups, -1, height, width)

        merged = cross_merge(ys, height, width).reshape(batch, -1, height, width)
        if not self.channel_first:
            merged = merged.permute(0, 2, 3, 1).contiguous()
        return self.out_norm(merged)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.in_proj(hidden_states)
        gate = None
        if self.use_gate:
            hidden_states, gate = hidden_states.chunk(2, dim=1 if self.channel_first else -1)
            gate = self.act(gate)
        if not self.channel_first:
            hidden_states = hidden_states.permute(0, 3, 1, 2).contiguous()
        if self.conv2d is not None:
            hidden_states = self.conv2d(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self._forward_core(hidden_states)
        if gate is not None:
            hidden_states = hidden_states * gate
        hidden_states = self.out_proj(hidden_states)
        return self.dropout(hidden_states)


class VMambaMLP(nn.Module):
    def __init__(self, config: VMambaConfig, dim: int) -> None:
        super().__init__()
        linear_cls = VMambaLinear2d if config.channel_first else nn.Linear
        hidden = int(dim * config.mlp_ratio)
        self.fc1 = linear_cls(dim, hidden)
        self.act = ACT2FN[config.mlp_act]
        self.fc2 = linear_cls(hidden, dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(hidden_states)))


class VMambaLayer(nn.Module):
    """One Visual State-Space block: LN → SS2D → residual, then LN → MLP → residual."""

    def __init__(self, config: VMambaConfig, dim: int, drop_path: float) -> None:
        super().__init__()
        norm_cls = VMambaLayerNorm2d if config.channel_first else nn.LayerNorm
        self.ssm_branch = config.ssm_ratio > 0
        self.mlp_branch = config.mlp_ratio > 0
        if self.ssm_branch:
            self.norm = norm_cls(dim, eps=config.layer_norm_eps)
            self.op = VMambaSS2D(config, dim)
        if self.mlp_branch:
            self.norm2 = norm_cls(dim, eps=config.layer_norm_eps)
            self.mlp = VMambaMLP(config, dim)
        self.drop_path = VMambaDropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.ssm_branch:
            hidden_states = hidden_states + self.drop_path(self.op(self.norm(hidden_states)))
        if self.mlp_branch:
            hidden_states = hidden_states + self.drop_path(self.mlp(self.norm2(hidden_states)))
        return hidden_states


class VMambaPatchEmbeddings(nn.Module):
    """Stem. `v1` is a single conv; `v2` (official) is two overlapping convs + GELU."""

    def __init__(self, config: VMambaConfig) -> None:
        super().__init__()
        self.num_channels = config.num_channels
        self.channel_first = config.channel_first
        embed_dim = config.hidden_sizes[0]
        patch_size = config.patch_size if isinstance(config.patch_size, int) else config.patch_size[0]
        norm_cls = VMambaLayerNorm2d if config.channel_first else nn.LayerNorm
        self.version = config.patch_embed_version

        if self.version == "v1":
            self.projection = nn.Conv2d(config.num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.norm = norm_cls(embed_dim, eps=config.layer_norm_eps) if config.patch_norm else nn.Identity()
            self.conv1 = self.norm1 = self.conv2 = self.norm2 = None
        elif self.version == "v2":
            stride = patch_size // 2
            kernel_size = stride + 1
            self.conv1 = nn.Conv2d(config.num_channels, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=1)
            self.norm1 = norm_cls(embed_dim // 2, eps=config.layer_norm_eps) if config.patch_norm else nn.Identity()
            self.act = nn.GELU()
            self.conv2 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=1)
            self.norm2 = norm_cls(embed_dim, eps=config.layer_norm_eps) if config.patch_norm else nn.Identity()
            self.projection = self.norm = None
        else:
            raise ValueError(f"Unknown patch_embed_version={self.version}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.shape[1] != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels, got {pixel_values.shape[1]}. "
                "Make sure pixel_values are in channels-first format."
            )
        if self.version == "v1":
            hidden_states = self.projection(pixel_values)
            if not self.channel_first:
                hidden_states = hidden_states.permute(0, 2, 3, 1)
            return self.norm(hidden_states)

        hidden_states = self.conv1(pixel_values)
        if not self.channel_first:
            hidden_states = hidden_states.permute(0, 2, 3, 1)
        hidden_states = self.norm1(hidden_states)
        if not self.channel_first:
            hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.act(hidden_states)
        hidden_states = self.conv2(hidden_states)
        if not self.channel_first:
            hidden_states = hidden_states.permute(0, 2, 3, 1)
        return self.norm2(hidden_states)


class VMambaDownsample(nn.Module):
    """Spatial downsampling between stages."""

    def __init__(self, config: VMambaConfig, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.version = config.downsample_version
        self.channel_first = config.channel_first
        norm_cls = VMambaLayerNorm2d if config.channel_first else nn.LayerNorm

        if self.version == "v1":
            linear_cls = VMambaLinear2d if config.channel_first else nn.Linear
            self.norm = norm_cls(4 * in_dim, eps=config.layer_norm_eps)
            self.reduction = linear_cls(4 * in_dim, out_dim, bias=False)
            self.conv = None
        elif self.version == "v2":
            self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
            self.norm = norm_cls(out_dim, eps=config.layer_norm_eps)
            self.reduction = None
        elif self.version == "v3":
            self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
            self.norm = norm_cls(out_dim, eps=config.layer_norm_eps)
            self.reduction = None
        else:
            raise ValueError(f"Unknown downsample_version={self.version}")

    def _merge_v1(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.channel_first:
            if hidden_states.shape[-1] % 2 or hidden_states.shape[-2] % 2:
                hidden_states = F.pad(hidden_states, (0, hidden_states.shape[-1] % 2, 0, hidden_states.shape[-2] % 2))
            x0 = hidden_states[..., 0::2, 0::2]
            x1 = hidden_states[..., 1::2, 0::2]
            x2 = hidden_states[..., 0::2, 1::2]
            x3 = hidden_states[..., 1::2, 1::2]
            return torch.cat([x0, x1, x2, x3], dim=1)
        if hidden_states.shape[2] % 2 or hidden_states.shape[1] % 2:
            hidden_states = F.pad(hidden_states, (0, 0, 0, hidden_states.shape[2] % 2, 0, hidden_states.shape[1] % 2))
        x0 = hidden_states[:, 0::2, 0::2, :]
        x1 = hidden_states[:, 1::2, 0::2, :]
        x2 = hidden_states[:, 0::2, 1::2, :]
        x3 = hidden_states[:, 1::2, 1::2, :]
        return torch.cat([x0, x1, x2, x3], dim=-1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.version == "v1":
            hidden_states = self._merge_v1(hidden_states)
            return self.reduction(self.norm(hidden_states))
        if not self.channel_first:
            hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.conv(hidden_states)
        if not self.channel_first:
            hidden_states = hidden_states.permute(0, 2, 3, 1)
        return self.norm(hidden_states)


class VMambaStage(nn.Module):
    """A hierarchical stage: a stack of VSS blocks and an optional downsample."""

    def __init__(
        self,
        config: VMambaConfig,
        dim: int,
        depth: int,
        drop_path_rates: list[float],
        downsample: Optional[nn.Module],
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [VMambaLayer(config, dim=dim, drop_path=drop_path_rates[i]) for i in range(depth)]
        )
        self.downsample = downsample if downsample is not None else nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            hidden_states = block(hidden_states)
        stage_output = hidden_states
        hidden_states = self.downsample(hidden_states)
        return stage_output, hidden_states


@auto_docstring
class VMambaPreTrainedModel(PreTrainedModel):
    config_class = VMambaConfig
    config: VMambaConfig
    base_model_prefix = "vmamba"
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    supports_gradient_checkpointing = True
    _no_split_modules = ["VMambaLayer", "VMambaStage"]
    _supports_sdpa = False

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, VMambaLinear2d, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, VMambaLayerNorm2d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class VMambaEncoder(nn.Module):
    def __init__(self, config: VMambaConfig) -> None:
        super().__init__()
        self.config = config
        dpr = torch.linspace(0, config.drop_path_rate, sum(config.depths)).tolist()
        self.layers = nn.ModuleList()
        cursor = 0
        for stage_idx, (dim, depth) in enumerate(zip(config.hidden_sizes, config.depths)):
            if stage_idx < config.num_stages - 1:
                downsample = VMambaDownsample(config, dim, config.hidden_sizes[stage_idx + 1])
            else:
                downsample = None
            self.layers.append(
                VMambaStage(
                    config,
                    dim=dim,
                    depth=depth,
                    drop_path_rates=dpr[cursor : cursor + depth],
                    downsample=downsample,
                )
            )
            cursor += depth

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> BaseModelOutputWithNoAttention:
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        for stage in self.layers:
            stage_output, hidden_states = stage(hidden_states)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (stage_output,)
        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=all_hidden_states)


@auto_docstring
class VMambaModel(VMambaPreTrainedModel):
    def __init__(self, config: VMambaConfig) -> None:
        super().__init__(config)
        self.config = config
        self.embeddings = VMambaPatchEmbeddings(config)
        self.encoder = VMambaEncoder(config)
        norm_cls = VMambaLayerNorm2d if config.channel_first else nn.LayerNorm
        self.layernorm = norm_cls(config.hidden_sizes[-1], eps=config.layer_norm_eps)
        self.post_init()

    def get_input_embeddings(self) -> VMambaPatchEmbeddings:
        return self.embeddings

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPoolingAndNoAttention:
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        embedding_output = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(embedding_output, output_hidden_states=output_hidden_states)
        last_hidden_state = self.layernorm(encoder_outputs.last_hidden_state)

        if self.config.channel_first:
            pooled_output = last_hidden_state.mean(dim=(-2, -1))
        else:
            pooled_output = last_hidden_state.mean(dim=(1, 2))

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


@auto_docstring(
    custom_intro="""
    VMamba Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """
)
class VMambaForImageClassification(VMambaPreTrainedModel):
    accepts_loss_kwargs = False

    def __init__(self, config: VMambaConfig) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.vmamba = VMambaModel(config)
        self.classifier = (
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity()
        )
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ImageClassifierOutputWithNoAttention:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs: BaseModelOutputWithPoolingAndNoAttention = self.vmamba(
            pixel_values, output_hidden_states=output_hidden_states, **kwargs
        )
        logits = self.classifier(outputs.pooler_output)

        loss = None
        if labels is not None:
            if hasattr(self, "loss_function"):
                loss = self.loss_function(labels=labels, pooled_logits=logits, config=self.config)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return ImageClassifierOutputWithNoAttention(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
        )


@auto_docstring(
    custom_intro="""
    VMamba backbone, to be used with frameworks like DETR, Mask R-CNN and UPerNet.
    """
)
class VMambaBackbone(BackboneMixin, VMambaPreTrainedModel):
    has_attentions = False

    def __init__(self, config: VMambaConfig) -> None:
        super().__init__(config)
        self.embeddings = VMambaPatchEmbeddings(config)
        self.encoder = VMambaEncoder(config)
        # stem + one entry per stage, matching `config.stage_names`
        self.num_features = [config.hidden_sizes[0]] + list(config.hidden_sizes)
        hidden_states_norms = {}
        norm_cls = VMambaLayerNorm2d if config.channel_first else nn.LayerNorm
        for stage, num_channels in zip(self.out_features, self.channels):
            hidden_states_norms[stage] = norm_cls(num_channels, eps=config.layer_norm_eps)
        self.hidden_states_norms = nn.ModuleDict(hidden_states_norms)
        self.post_init()

    @can_return_tuple
    @filter_output_hidden_states
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = True,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BackboneOutput:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, AutoBackbone
        >>> import torch
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> processor = AutoImageProcessor.from_pretrained("MzeroMiko/vmamba-tiny-s1l8")
        >>> model = AutoBackbone.from_pretrained("MzeroMiko/vmamba-tiny-s1l8")

        >>> inputs = processor(image, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```"""
        embedding_output = self.embeddings(pixel_values)
        encoder_outputs = self.encoder(embedding_output, output_hidden_states=True)
        hidden_states = encoder_outputs.hidden_states

        feature_maps = []
        # hidden_states[0] is stem; hidden_states[i] is stage i (pre-downsample)
        for stage, hidden_state in zip(self.stage_names, hidden_states):
            if stage in self.out_features:
                hidden_state = self.hidden_states_norms[stage](hidden_state)
                if not self.config.channel_first:
                    hidden_state = hidden_state.permute(0, 3, 1, 2)
                feature_maps.append(hidden_state)

        return BackboneOutput(feature_maps=tuple(feature_maps), hidden_states=hidden_states)


__all__ = [
    "VMambaPreTrainedModel",
    "VMambaModel",
    "VMambaForImageClassification",
    "VMambaBackbone",
    "VMAMBA_PRETRAINED_MODEL_ARCHIVE_LIST",
]
