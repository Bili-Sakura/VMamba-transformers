# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Convert an official MzeroMiko/VMamba checkpoint to Hugging Face format.

Official checkpoints (classification):
- VMamba-T s1l8: https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm1_tiny_0230s_ckpt_epoch_264.pth
- VMamba-S s2l15: https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_small_0229_ckpt_epoch_222.pth
- VMamba-B s2l15: https://github.com/MzeroMiko/VMamba/releases/download/%23v2cls/vssm_base_0229_ckpt_epoch_237.pth

Example:
    python -m vmamba.convert_vmamba_original_to_hf \\
        --variant tiny --pytorch_dump_folder_path ./vmamba-tiny-s1l8 \\
        --original_checkpoint /path/to/vssm1_tiny_0230s_ckpt_epoch_264.pth
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from .configuration_vmamba import VMambaConfig
from .modeling_vmamba import VMambaForImageClassification
from .pipeline import pipeline as build_pipeline
from .pipeline import save_pretrained as save_pipeline


VARIANT_TO_CONFIG = {
    "tiny": VMambaConfig.vmamba_tiny,
    "small": VMambaConfig.vmamba_small,
    "base": VMambaConfig.vmamba_base,
}


def _rename_key(key: str) -> str:
    """Map an official VSSM state-dict key to the Hugging Face module tree."""
    # strip common wrappers
    for prefix in ("module.", "model.", "backbone."):
        if key.startswith(prefix):
            key = key[len(prefix) :]

    # stem (patch embed v2, channel-first): Sequential indices 0,2,5,7
    key = key.replace("patch_embed.0.", "embeddings.conv1.")
    key = key.replace("patch_embed.2.", "embeddings.norm1.")
    key = key.replace("patch_embed.5.", "embeddings.conv2.")
    key = key.replace("patch_embed.7.", "embeddings.norm2.")
    # stem v1
    key = key.replace("patch_embed.proj.", "embeddings.projection.")
    key = key.replace("patch_embed.norm.", "embeddings.norm.")

    # stages
    key = key.replace("layers.", "encoder.layers.")

    # downsample v2/v3, channel-first Sequential: Identity, Conv, Identity, Norm
    key = re.sub(r"downsample\.1\.", "downsample.conv.", key)
    key = re.sub(r"downsample\.3\.", "downsample.norm.", key)
    # downsample v1
    key = key.replace("downsample.reduction.", "downsample.reduction.")
    key = key.replace("downsample.norm.", "downsample.norm.")

    # classifier
    if key.startswith("classifier.norm"):
        key = key.replace("classifier.norm", "layernorm")
    if key.startswith("classifier.head"):
        key = key.replace("classifier.head", "classifier")
    if key.startswith("head."):
        key = key.replace("head.", "classifier.")
    if key.startswith("norm.") and not key.startswith("encoder."):
        key = key.replace("norm.", "layernorm.", 1)

    # wrap backbone weights
    if not key.startswith("classifier"):
        key = "vmamba." + key
    return key


def convert_state_dict(state_dict: dict) -> dict:
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    converted = {}
    for old_key, value in state_dict.items():
        if old_key.endswith("total_ops") or old_key.endswith("total_params"):
            continue
        converted[_rename_key(old_key)] = value
    return converted


@torch.no_grad()
def convert_vmamba_checkpoint(
    original_checkpoint: str | None,
    pytorch_dump_folder_path: str,
    variant: str = "tiny",
    num_labels: int = 1000,
) -> VMambaForImageClassification:
    if variant not in VARIANT_TO_CONFIG:
        raise ValueError(f"Unknown variant {variant}. Choose from {list(VARIANT_TO_CONFIG)}")

    config = VARIANT_TO_CONFIG[variant](num_labels=num_labels)
    config.id2label = {i: str(i) for i in range(num_labels)}
    config.label2id = {str(i): i for i in range(num_labels)}

    model = VMambaForImageClassification(config)
    model.eval()

    if original_checkpoint is not None:
        checkpoint = torch.load(original_checkpoint, map_location="cpu", weights_only=False)
        converted = convert_state_dict(checkpoint)
        missing, unexpected = model.load_state_dict(converted, strict=False)
        print(f"Loaded {original_checkpoint}")
        print(f"missing keys ({len(missing)}): {missing[:20]}{'...' if len(missing) > 20 else ''}")
        print(f"unexpected keys ({len(unexpected)}): {unexpected[:20]}{'...' if len(unexpected) > 20 else ''}")

    dump_path = Path(pytorch_dump_folder_path)
    dump_path.mkdir(parents=True, exist_ok=True)
    pipe = build_pipeline(task="image-classification", model=model, device=-1)
    save_pipeline(dump_path, pipe)
    print(f"Saved custom pipeline checkpoint to {dump_path}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original_checkpoint", type=str, default=None, help="Path to a .pth official checkpoint.")
    parser.add_argument("--pytorch_dump_folder_path", type=str, required=True, help="Output directory.")
    parser.add_argument("--variant", type=str, default="tiny", choices=sorted(VARIANT_TO_CONFIG))
    parser.add_argument("--num_labels", type=int, default=1000)
    args = parser.parse_args()
    convert_vmamba_checkpoint(
        args.original_checkpoint, args.pytorch_dump_folder_path, args.variant, args.num_labels
    )


if __name__ == "__main__":
    main()


__all__ = ["convert_state_dict", "convert_vmamba_checkpoint", "VARIANT_TO_CONFIG"]
