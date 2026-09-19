# Copyright 2024 The HuggingFace Team and VMamba authors. All rights reserved.
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
"""Standalone Hugging Face–style VMamba package.

Uses `transformers` as a library (PreTrainedModel, Pipeline) but does **not**
patch or PR into official huggingface/transformers. Inference entry point is
the custom pipeline in [`pipeline`][vmamba.pipeline.pipeline].
"""

from .configuration_vmamba import VMAMBA_PRETRAINED_CONFIG_ARCHIVE_MAP, VMambaConfig
from .image_processing_vmamba import VMambaImageProcessor
from .modeling_vmamba import (
    VMAMBA_PRETRAINED_MODEL_ARCHIVE_LIST,
    VMambaBackbone,
    VMambaForImageClassification,
    VMambaModel,
    VMambaPreTrainedModel,
)
from .pipeline import (
    VMambaFeatureExtractionPipeline,
    VMambaImageClassificationPipeline,
    pipeline,
    save_pretrained,
)

__all__ = [
    "VMambaConfig",
    "VMambaModel",
    "VMambaForImageClassification",
    "VMambaBackbone",
    "VMambaPreTrainedModel",
    "VMambaImageProcessor",
    "VMambaImageClassificationPipeline",
    "VMambaFeatureExtractionPipeline",
    "pipeline",
    "save_pretrained",
    "VMAMBA_PRETRAINED_CONFIG_ARCHIVE_MAP",
    "VMAMBA_PRETRAINED_MODEL_ARCHIVE_LIST",
]
