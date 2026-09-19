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
"""Image processor class for VMamba.

Matches the original ImageNet evaluation recipe used by VMamba / Swin:
resize the shorter edge to ``224 / crop_pct`` (256 when ``crop_pct = 224/256``),
center-crop to 224, rescale to ``[0, 1]``, and normalize with ImageNet mean/std.
"""

from __future__ import annotations

from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
from transformers.models.convnext.image_processing_convnext import ConvNextImageProcessor
from transformers.utils import auto_docstring


@auto_docstring
class VMambaImageProcessor(ConvNextImageProcessor):
    """ImageNet-style processor for VMamba (bicubic resize + center crop)."""

    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"shortest_edge": 224}
    default_to_square = False
    do_resize = True
    do_rescale = True
    do_normalize = True
    crop_pct = 224 / 256


__all__ = ["VMambaImageProcessor"]
