# Copyright 2024 The HuggingFace Inc. team and VMamba authors. All rights reserved.
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
"""Custom Hugging Face pipelines for VMamba.

This is a *standalone* custom pipeline (not part of official `transformers`).
Use it as:

```python
from vmamba import pipeline

pipe = pipeline(variant="tiny")          # randomly initialized
pipe = pipeline(model="./vmamba-tiny")   # converted checkpoint
print(pipe("cat.jpg", top_k=5))
```

Saved checkpoints include `auto_map` + `custom_pipelines` so a Hub repo that
also ships these `.py` files can be loaded with
`transformers.pipeline(..., trust_remote_code=True)`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
from PIL import Image
from transformers.image_utils import load_image
from transformers.pipelines import Pipeline

from .configuration_vmamba import VMambaConfig
from .image_processing_vmamba import VMambaImageProcessor
from .modeling_vmamba import VMambaForImageClassification, VMambaModel


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.exp(values - np.max(values, axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


class VMambaImageClassificationPipeline(Pipeline):
    """Image classification pipeline for [`VMambaForImageClassification`].

    Accepts a local path, a URL, or a PIL image (or a batch of those) and
    returns `[{"label": ..., "score": ...}, ...]` like the official
    `image-classification` pipeline.
    """

    _load_processor = False
    _load_image_processor = True
    _load_feature_extractor = False
    _load_tokenizer = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.image_processor is None:
            self.image_processor = VMambaImageProcessor()

    def _sanitize_parameters(self, top_k=None, function_to_apply=None, timeout=None):
        preprocess_params = {}
        if timeout is not None:
            preprocess_params["timeout"] = timeout
        postprocess_params = {}
        if top_k is not None:
            postprocess_params["top_k"] = top_k
        if function_to_apply is not None:
            postprocess_params["function_to_apply"] = function_to_apply
        return preprocess_params, {}, postprocess_params

    def __call__(
        self,
        inputs: Union[str, list[str], "Image.Image", list["Image.Image"], None] = None,
        **kwargs: Any,
    ):
        if "images" in kwargs:
            inputs = kwargs.pop("images")
        if inputs is None:
            raise ValueError("Pass an image path, URL, PIL image, or a list of those.")
        return super().__call__(inputs, **kwargs)

    def preprocess(self, image, timeout=None):
        if not isinstance(image, (np.ndarray, torch.Tensor)):
            image = load_image(image, timeout=timeout)
        model_inputs = self.image_processor(images=image, return_tensors="pt")
        if hasattr(model_inputs, "to"):
            model_inputs = model_inputs.to(getattr(self, "dtype", torch.float32))
        return model_inputs

    def _forward(self, model_inputs):
        return self.model(**model_inputs)

    def postprocess(self, model_outputs, function_to_apply=None, top_k=5):
        num_labels = int(getattr(self.model.config, "num_labels", 0) or 0)
        if function_to_apply is None:
            function_to_apply = "sigmoid" if num_labels == 1 else "softmax"
        if top_k is None or (num_labels and top_k > num_labels):
            top_k = max(num_labels, 1)

        logits = model_outputs["logits"][0]
        if logits.dtype in (torch.bfloat16, torch.float16):
            logits = logits.to(torch.float32)
        scores = logits.detach().cpu().numpy()
        if function_to_apply == "softmax":
            scores = _softmax(scores)
        elif function_to_apply == "sigmoid":
            scores = _sigmoid(scores)
        elif function_to_apply != "none":
            raise ValueError(f"Unrecognized function_to_apply={function_to_apply}")

        id2label = getattr(self.model.config, "id2label", None) or {}
        results = [
            {"label": id2label.get(i, id2label.get(str(i), str(i))), "score": float(score)}
            for i, score in enumerate(scores)
        ]
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]


class VMambaFeatureExtractionPipeline(Pipeline):
    """Return pooled VMamba embeddings for an image (or a batch)."""

    _load_processor = False
    _load_image_processor = True
    _load_feature_extractor = False
    _load_tokenizer = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.image_processor is None:
            self.image_processor = VMambaImageProcessor()

    def _sanitize_parameters(self, timeout=None, return_tensors=None):
        preprocess_params = {}
        if timeout is not None:
            preprocess_params["timeout"] = timeout
        postprocess_params = {}
        if return_tensors is not None:
            postprocess_params["return_tensors"] = return_tensors
        return preprocess_params, {}, postprocess_params

    def preprocess(self, image, timeout=None):
        if not isinstance(image, (np.ndarray, torch.Tensor)):
            image = load_image(image, timeout=timeout)
        model_inputs = self.image_processor(images=image, return_tensors="pt")
        if hasattr(model_inputs, "to"):
            model_inputs = model_inputs.to(getattr(self, "dtype", torch.float32))
        return model_inputs

    def _forward(self, model_inputs):
        outputs = self.model(**model_inputs)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state.mean(dim=(-2, -1))
        return pooled

    def postprocess(self, model_outputs, return_tensors=False):
        if return_tensors:
            return model_outputs
        values = model_outputs.detach().cpu()
        if values.ndim > 1 and values.shape[0] == 1:
            values = values[0]
        return values.tolist()


_VARIANT_BUILDERS = {
    "tiny": VMambaConfig.vmamba_tiny,
    "small": VMambaConfig.vmamba_small,
    "base": VMambaConfig.vmamba_base,
}

_REMOTE_CODE_FILES = (
    "configuration_vmamba.py",
    "modeling_vmamba.py",
    "image_processing_vmamba.py",
    "pipeline.py",
    "__init__.py",
)


def _build_model(variant: str, num_labels: int, task: str):
    if variant not in _VARIANT_BUILDERS:
        raise ValueError(f"Unknown variant {variant!r}. Choose from {sorted(_VARIANT_BUILDERS)}.")
    config = _VARIANT_BUILDERS[variant](num_labels=num_labels)
    if not config.id2label:
        config.id2label = {i: str(i) for i in range(num_labels)}
        config.label2id = {str(i): i for i in range(num_labels)}
    if task == "feature-extraction":
        return VMambaModel(config)
    return VMambaForImageClassification(config)


def pipeline(
    task: str = "image-classification",
    model: str | Path | VMambaForImageClassification | VMambaModel | None = None,
    image_processor: VMambaImageProcessor | None = None,
    variant: str = "tiny",
    num_labels: int = 1000,
    device: int | str | torch.device | None = None,
    **kwargs,
) -> Pipeline:
    """Build a VMamba custom pipeline.

    Args:
        task: `"image-classification"` or `"feature-extraction"`.
        model: A converted checkpoint directory, a model instance, or `None`
            to construct a randomly initialized official variant.
        image_processor: Optional processor. Loaded from `model` when that is a
            directory, otherwise a default ImageNet processor is used.
        variant: `"tiny"`, `"small"`, or `"base"` when `model` is `None`.
        num_labels: Classification head size for a randomly initialized model.
        device: Device index or name. Defaults to CUDA if available.
    """
    if image_processor is None:
        if isinstance(model, (str, Path)) and Path(model).exists():
            try:
                image_processor = VMambaImageProcessor.from_pretrained(model)
            except Exception:
                image_processor = VMambaImageProcessor()
        else:
            image_processor = VMambaImageProcessor()

    if model is None:
        model = _build_model(variant, num_labels, task)
    elif isinstance(model, (str, Path)):
        model_cls = VMambaModel if task == "feature-extraction" else VMambaForImageClassification
        model = model_cls.from_pretrained(model)

    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    if task in {"image-classification", "vmamba-image-classification"}:
        return VMambaImageClassificationPipeline(
            model=model,
            image_processor=image_processor,
            task="image-classification",
            device=device,
            **kwargs,
        )
    if task in {"feature-extraction", "vmamba-feature-extraction"}:
        if isinstance(model, VMambaForImageClassification):
            model = model.vmamba
        return VMambaFeatureExtractionPipeline(
            model=model,
            image_processor=image_processor,
            task="feature-extraction",
            device=device,
            **kwargs,
        )
    raise ValueError(
        f"Unsupported task {task!r}. Use 'image-classification' or 'feature-extraction'."
    )


def save_pretrained(save_directory: str | Path, pipe: Pipeline) -> Path:
    """Save a pipeline so it can be reloaded with [`pipeline`] or Hub `trust_remote_code`.

    Copies the custom-code modules next to `config.json` and writes `auto_map`
    / `custom_pipelines` entries. Official `transformers` does not need a PR.
    """
    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)

    pipe.model.save_pretrained(save_directory)
    if pipe.image_processor is not None:
        pipe.image_processor.save_pretrained(save_directory)

    package_dir = Path(__file__).resolve().parent
    for filename in _REMOTE_CODE_FILES:
        src = package_dir / filename
        if src.exists():
            shutil.copy2(src, save_directory / filename)

    config_path = save_directory / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    config["model_type"] = "vmamba"
    config["auto_map"] = {
        "AutoConfig": "configuration_vmamba.VMambaConfig",
        "AutoModel": "modeling_vmamba.VMambaModel",
        "AutoModelForImageClassification": "modeling_vmamba.VMambaForImageClassification",
        "AutoImageProcessor": "image_processing_vmamba.VMambaImageProcessor",
    }
    config["custom_pipelines"] = {
        "image-classification": {
            "impl": "pipeline.VMambaImageClassificationPipeline",
            "pt": ["VMambaForImageClassification"],
            "type": "image",
        },
        "feature-extraction": {
            "impl": "pipeline.VMambaFeatureExtractionPipeline",
            "pt": ["VMambaModel"],
            "type": "image",
        },
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return save_directory


__all__ = [
    "VMambaImageClassificationPipeline",
    "VMambaFeatureExtractionPipeline",
    "pipeline",
    "save_pretrained",
]
