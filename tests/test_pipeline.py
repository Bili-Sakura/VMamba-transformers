# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
"""Tests for the standalone VMamba custom pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vmamba import (
    VMambaConfig,
    VMambaFeatureExtractionPipeline,
    VMambaForImageClassification,
    VMambaImageClassificationPipeline,
    VMambaImageProcessor,
    pipeline,
    save_pretrained,
)


def _tiny_model(num_labels: int = 4) -> VMambaForImageClassification:
    config = VMambaConfig(
        image_size=32,
        patch_size=4,
        depths=(1, 1),
        embed_dim=16,
        ssm_state_size=2,
        ssm_ratio=1.0,
        mlp_ratio=2.0,
        drop_path_rate=0.0,
        num_labels=num_labels,
        id2label={i: f"class_{i}" for i in range(num_labels)},
        label2id={f"class_{i}": i for i in range(num_labels)},
    )
    model = VMambaForImageClassification(config)
    model.eval()
    return model


def _dummy_image(size: int = 40) -> Image.Image:
    array = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(array)


def test_classification_pipeline_on_pil_image():
    model = _tiny_model()
    processor = VMambaImageProcessor(size={"shortest_edge": 32}, crop_pct=1.0)
    pipe = pipeline(
        task="image-classification",
        model=model,
        image_processor=processor,
        device=-1,
    )
    assert isinstance(pipe, VMambaImageClassificationPipeline)
    results = pipe(_dummy_image(), top_k=3)
    assert len(results) == 3
    assert {"label", "score"} <= set(results[0])
    assert results[0]["score"] >= results[-1]["score"]
    assert results[0]["label"].startswith("class_")


def test_classification_pipeline_batch():
    model = _tiny_model()
    processor = VMambaImageProcessor(size={"shortest_edge": 32}, crop_pct=1.0)
    pipe = pipeline(model=model, image_processor=processor, device=-1)
    outputs = pipe([_dummy_image(), _dummy_image()], top_k=2)
    assert len(outputs) == 2
    assert all(len(item) == 2 for item in outputs)


def test_feature_extraction_pipeline():
    model = _tiny_model()
    processor = VMambaImageProcessor(size={"shortest_edge": 32}, crop_pct=1.0)
    pipe = pipeline(
        task="feature-extraction",
        model=model,
        image_processor=processor,
        device=-1,
    )
    assert isinstance(pipe, VMambaFeatureExtractionPipeline)
    features = pipe(_dummy_image())
    assert isinstance(features, list)
    assert len(features) == model.config.hidden_sizes[-1]


def test_pipeline_from_variant():
    pipe = pipeline(variant="tiny", num_labels=10, device=-1)
    assert pipe.model.config.num_labels == 10
    assert pipe.model.config.depths == (2, 2, 8, 2)


def test_save_and_reload_pipeline(tmp_path: Path):
    model = _tiny_model()
    processor = VMambaImageProcessor(size={"shortest_edge": 32}, crop_pct=1.0)
    pipe = pipeline(model=model, image_processor=processor, device=-1)
    image = _dummy_image()
    before = pipe(image, top_k=4)

    save_pretrained(tmp_path, pipe)
    assert (tmp_path / "pipeline.py").exists()
    assert (tmp_path / "modeling_vmamba.py").exists()
    config = json.loads((tmp_path / "config.json").read_text())
    assert "custom_pipelines" in config
    assert "auto_map" in config

    reloaded = pipeline(model=tmp_path, device=-1)
    after = reloaded(image, top_k=4)
    assert [item["label"] for item in before] == [item["label"] for item in after]
    assert all(abs(a["score"] - b["score"]) < 1e-4 for a, b in zip(before, after))


def test_pipeline_rejects_unknown_task():
    with pytest.raises(ValueError, match="Unsupported task"):
        pipeline(task="text-generation", model=_tiny_model(), device=-1)


def test_pipeline_requires_input():
    pipe = pipeline(model=_tiny_model(), image_processor=VMambaImageProcessor(), device=-1)
    with pytest.raises(ValueError):
        pipe(None)
