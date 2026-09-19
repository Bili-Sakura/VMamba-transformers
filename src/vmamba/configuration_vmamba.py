# Copyright 2024 MzeroMiko, Derk Mus, and The HuggingFace Inc. team. All rights reserved.
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
"""VMamba model configuration."""

from __future__ import annotations

import math

from transformers.backbone_utils import BackboneConfigMixin
from transformers.configuration_utils import PreTrainedConfig
from transformers.utils import auto_docstring

try:
    from huggingface_hub.dataclasses import strict
except ImportError:  # transformers < 5
    def strict(cls):
        return cls


VMAMBA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # Official ImageNet-1K checkpoints from https://github.com/MzeroMiko/VMamba
    # Convert with `convert_vmamba_original_to_hf.py` then push to the Hub.
}


def _hidden_sizes_from_embed_dim(embed_dim: int, num_stages: int) -> tuple[int, ...]:
    return tuple(int(embed_dim * (2**i)) for i in range(num_stages))


@auto_docstring(checkpoint="MzeroMiko/vmamba-tiny-s1l8")
@strict
class VMambaConfig(BackboneConfigMixin, PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VMambaModel`]. It is used to instantiate a
    VMamba model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of VMamba-Tiny s1l8
    ([`MzeroMiko/VMamba`](https://github.com/MzeroMiko/VMamba)).

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        image_size (`int`, *optional*, defaults to 224):
            The expected input image size.
        patch_size (`int`, *optional*, defaults to 4):
            The size (resolution) of each stem patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        embed_dim (`int`, *optional*, defaults to 96):
            Patch embedding dimension of the first stage. Stage widths are `embed_dim * 2**i` unless `hidden_sizes`
            is provided.
        depths (`list[int]`, *optional*, defaults to `(2, 2, 8, 2)`):
            Number of VSS blocks in each stage. Default is VMamba-Tiny s1l8.
        hidden_sizes (`list[int]`, *optional*):
            Dimensionality of each stage. If unset, derived from `embed_dim`.
        ssm_state_size (`int`, *optional*, defaults to 1):
            State size `N` of the selective SSM (official v2 models use `1`).
        ssm_ratio (`float`, *optional*, defaults to 1.0):
            Expansion ratio used to form the SSM inner dimension `d_inner = ssm_ratio * hidden_size`.
        ssm_dt_rank (`int` or `"auto"`, *optional*, defaults to `"auto"`):
            Rank of the Δ (time-step) projection. `"auto"` uses `ceil(hidden_size / 16)`.
        ssm_conv_kernel (`int`, *optional*, defaults to 3):
            Kernel size of the depthwise conv before the scan. Values `< 2` disable the conv.
        ssm_conv_bias (`bool`, *optional*, defaults to `False`):
            Whether the depthwise conv has a bias (official v2: `False`).
        ssm_use_gate (`bool`, *optional*, defaults to `False`):
            If `True`, SS2D uses a SiLU-gated branch (`z`). Official v2 uses `v05_noz` (`False`).
        ssm_init (`str`, *optional*, defaults to `"v0"`):
            Initialization scheme for `A`, `D` and Δ (`"v0"`, `"v1"`, or `"v2"`).
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            Expansion ratio of the FFN. Set to `0` to disable the MLP branch (vanilla VMamba v0).
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            Activation used inside SS2D.
        mlp_act (`str`, *optional*, defaults to `"gelu"`):
            Activation used inside the MLP.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            Dropout applied after the SS2D output projection.
        drop_path_rate (`float`, *optional*, defaults to 0.2):
            Stochastic depth rate.
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            The epsilon used by layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        patch_norm (`bool`, *optional*, defaults to `True`):
            Whether to apply LayerNorm after the stem.
        channel_first (`bool`, *optional*, defaults to `True`):
            If `True`, activations are stored as `(batch, channels, height, width)` (official v2 `ln2d`).
        patch_embed_version (`str`, *optional*, defaults to `"v2"`):
            Stem type. `"v1"` is a single stride-`patch_size` conv; `"v2"` is two overlapping convs (official v2).
        downsample_version (`str`, *optional*, defaults to `"v3"`):
            Downsampling type between stages: `"v1"` patch-merging, `"v2"` 2x2 conv, `"v3"` 3x3 stride-2 conv.
        use_bias (`bool`, *optional*, defaults to `False`):
            Whether SS2D `in_proj` / `out_proj` use bias.
        use_associative_scan (`bool`, *optional*, defaults to `True`):
            Use a vectorized parallel prefix scan instead of a Python loop. The loop is only used as a reference.

    Example:
        ```python
        >>> from transformers import VMambaConfig, VMambaModel

        >>> # Initializing a VMamba-Tiny s1l8 style configuration
        >>> configuration = VMambaConfig()

        >>> # Initializing a model (with random weights) from the configuration
        >>> model = VMambaModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
    """

    model_type = "vmamba"
    attribute_map = {
        "num_hidden_layers": "num_stages",
    }

    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 4
    num_channels: int = 3
    embed_dim: int = 96
    depths: list[int] | tuple[int, ...] = (2, 2, 8, 2)
    hidden_sizes: list[int] | tuple[int, ...] | None = None
    ssm_state_size: int = 1
    ssm_ratio: float | int = 1.0
    ssm_dt_rank: str | int = "auto"
    ssm_conv_kernel: int = 3
    ssm_conv_bias: bool = False
    ssm_use_gate: bool = False
    ssm_init: str = "v0"
    mlp_ratio: float | int = 4.0
    hidden_act: str = "silu"
    mlp_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    drop_path_rate: float | int = 0.2
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    patch_norm: bool = True
    channel_first: bool = True
    patch_embed_version: str = "v2"
    downsample_version: str = "v3"
    use_bias: bool = False
    use_associative_scan: bool = True
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    def __post_init__(self, **kwargs):
        if self.hidden_sizes is None:
            self.hidden_sizes = _hidden_sizes_from_embed_dim(self.embed_dim, len(self.depths))
        else:
            self.hidden_sizes = tuple(self.hidden_sizes)
        self.depths = tuple(self.depths)
        if len(self.hidden_sizes) != len(self.depths):
            raise ValueError(
                f"`hidden_sizes` ({len(self.hidden_sizes)}) and `depths` ({len(self.depths)}) must have the same length."
            )
        self.num_stages = len(self.depths)
        self.hidden_size = int(self.hidden_sizes[-1])
        if self.ssm_dt_rank == "auto":
            self.ssm_dt_rank = math.ceil(self.embed_dim / 16)
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, self.num_stages + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None),
            out_features=kwargs.pop("out_features", None),
        )
        super().__post_init__(**kwargs)

    @classmethod
    def vmamba_tiny(cls, **kwargs):
        """VMamba-T[`s1l8`] — 30M / 4.9G, 82.6% ImageNet-1K."""
        defaults = dict(
            depths=(2, 2, 8, 2),
            embed_dim=96,
            ssm_state_size=1,
            ssm_ratio=1.0,
            ssm_conv_bias=False,
            ssm_use_gate=False,
            mlp_ratio=4.0,
            drop_path_rate=0.2,
            patch_embed_version="v2",
            downsample_version="v3",
            channel_first=True,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def vmamba_small(cls, **kwargs):
        """VMamba-S[`s2l15`] — 50M / 8.7G, 83.6% ImageNet-1K."""
        defaults = dict(
            depths=(2, 2, 15, 2),
            embed_dim=96,
            ssm_state_size=1,
            ssm_ratio=2.0,
            ssm_conv_bias=False,
            ssm_use_gate=False,
            mlp_ratio=4.0,
            drop_path_rate=0.3,
            patch_embed_version="v2",
            downsample_version="v3",
            channel_first=True,
        )
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def vmamba_base(cls, **kwargs):
        """VMamba-B[`s2l15`] — 89M / 15.4G, 83.9% ImageNet-1K."""
        defaults = dict(
            depths=(2, 2, 15, 2),
            embed_dim=128,
            ssm_state_size=1,
            ssm_ratio=2.0,
            ssm_conv_bias=False,
            ssm_use_gate=False,
            mlp_ratio=4.0,
            drop_path_rate=0.6,
            patch_embed_version="v2",
            downsample_version="v3",
            channel_first=True,
        )
        defaults.update(kwargs)
        return cls(**defaults)


__all__ = ["VMambaConfig", "VMAMBA_PRETRAINED_CONFIG_ARCHIVE_MAP"]
