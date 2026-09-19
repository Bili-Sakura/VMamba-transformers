# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
"""Tests for the standalone VMamba model (no official transformers integration)."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch

from vmamba import VMambaBackbone, VMambaConfig, VMambaForImageClassification, VMambaImageProcessor, VMambaModel
from vmamba.modeling_vmamba import _parallel_prefix_scan, _sequential_scan, cross_merge, cross_scan


def _tiny_config(**kwargs) -> VMambaConfig:
    defaults = dict(
        image_size=32,
        patch_size=4,
        depths=(1, 1),
        embed_dim=16,
        ssm_state_size=2,
        ssm_ratio=1.0,
        ssm_conv_bias=False,
        ssm_use_gate=False,
        mlp_ratio=2.0,
        drop_path_rate=0.0,
        hidden_dropout_prob=0.0,
        patch_embed_version="v2",
        downsample_version="v3",
        channel_first=True,
        num_labels=5,
    )
    defaults.update(kwargs)
    return VMambaConfig(**defaults)


@pytest.fixture
def tiny_config() -> VMambaConfig:
    return _tiny_config()


def test_config_defaults_match_official_tiny():
    config = VMambaConfig()
    assert config.depths == (2, 2, 8, 2)
    assert config.hidden_sizes == (96, 192, 384, 768)
    assert config.ssm_state_size == 1
    assert config.ssm_ratio == 1.0
    assert config.ssm_use_gate is False
    assert config.channel_first is True
    assert config.model_type == "vmamba"


def test_variant_factories():
    tiny = VMambaConfig.vmamba_tiny()
    small = VMambaConfig.vmamba_small()
    base = VMambaConfig.vmamba_base()
    assert tiny.hidden_sizes == (96, 192, 384, 768)
    assert small.depths == (2, 2, 15, 2)
    assert small.ssm_ratio == 2.0
    assert base.embed_dim == 128
    assert base.hidden_sizes == (128, 256, 512, 1024)


def test_forward_shapes(tiny_config):
    torch.manual_seed(0)
    model = VMambaForImageClassification(tiny_config)
    model.eval()
    pixel_values = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        outputs = model(pixel_values)
    assert outputs.logits.shape == (2, tiny_config.num_labels)

    backbone = VMambaModel(tiny_config)
    backbone.eval()
    with torch.no_grad():
        outputs = backbone(pixel_values)
    assert outputs.pooler_output.shape == (2, tiny_config.hidden_sizes[-1])
    assert outputs.last_hidden_state.shape[0] == 2
    assert outputs.last_hidden_state.shape[1] == tiny_config.hidden_sizes[-1]


def test_classification_loss(tiny_config):
    model = VMambaForImageClassification(tiny_config)
    model.train()
    pixel_values = torch.randn(3, 3, 32, 32)
    labels = torch.tensor([0, 1, 2])
    outputs = model(pixel_values, labels=labels)
    assert outputs.loss is not None
    outputs.loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_hidden_states(tiny_config):
    model = VMambaModel(tiny_config)
    model.eval()
    pixel_values = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        outputs = model(pixel_values, output_hidden_states=True)
    assert outputs.hidden_states is not None
    assert len(outputs.hidden_states) == 1 + tiny_config.num_stages


def test_forward_signature():
    for cls in (VMambaModel, VMambaForImageClassification):
        signature = inspect.signature(cls.forward)
        assert list(signature.parameters)[1] == "pixel_values"


def test_save_and_load(tiny_config, tmp_path: Path):
    model = VMambaForImageClassification(tiny_config)
    model.eval()
    pixel_values = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        logits_before = model(pixel_values).logits

    model.save_pretrained(tmp_path)
    loaded = VMambaForImageClassification.from_pretrained(tmp_path)
    loaded.eval()
    with torch.no_grad():
        logits_after = loaded(pixel_values).logits
    assert torch.allclose(logits_before, logits_after, atol=1e-5)


def test_backbone_feature_maps(tiny_config):
    tiny_config.out_indices = [1, 2]
    model = VMambaBackbone(tiny_config)
    model.eval()
    pixel_values = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        outputs = model(pixel_values)
    assert len(outputs.feature_maps) == 2
    assert all(t.ndim == 4 for t in outputs.feature_maps)
    assert outputs.feature_maps[0].shape[1] == tiny_config.hidden_sizes[0]
    assert outputs.feature_maps[1].shape[1] == tiny_config.hidden_sizes[1]


def test_greyscale_rejected_by_default(tiny_config):
    model = VMambaModel(tiny_config)
    with pytest.raises(ValueError):
        model(torch.randn(1, 1, 32, 32))


def test_cross_scan_merge_roundtrip():
    torch.manual_seed(0)
    hidden = torch.randn(2, 8, 6, 7)
    scanned = cross_scan(hidden)
    merged = cross_merge(scanned.reshape(2, 4, 8, 6, 7), 6, 7)
    assert merged.shape == (2, 8, 6 * 7)
    assert torch.isfinite(merged).all()
    assert torch.allclose(scanned[:, 0], hidden.flatten(2, 3))
    assert torch.allclose(scanned[:, 2], hidden.flatten(2, 3).flip(-1))


def test_parallel_scan_matches_sequential():
    torch.manual_seed(0)
    batch, dim, length, state = 2, 4, 17, 3
    delta_a = torch.rand(batch, dim, length, state) * 0.5 + 0.5
    delta_b_u = torch.randn(batch, dim, length, state)
    parallel = _parallel_prefix_scan(delta_a, delta_b_u)
    sequential = _sequential_scan(delta_a, delta_b_u)
    assert torch.allclose(parallel, sequential, atol=1e-5, rtol=1e-4)


def test_image_processor_numpy():
    processor = VMambaImageProcessor()
    image = torch.randint(0, 255, (32, 32, 3), dtype=torch.uint8).numpy()
    encoded = processor(image, return_tensors="pt", do_resize=False, do_center_crop=False)
    assert "pixel_values" in encoded
    assert encoded.pixel_values.ndim == 4
    assert encoded.pixel_values.shape[1] == 3


def test_key_conversion_roundtrip_names():
    from vmamba.convert_vmamba_original_to_hf import convert_state_dict

    original = {
        "patch_embed.0.weight": torch.zeros(1),
        "patch_embed.2.weight": torch.zeros(1),
        "layers.0.blocks.0.op.in_proj.weight": torch.zeros(1),
        "layers.0.downsample.1.weight": torch.zeros(1),
        "classifier.norm.weight": torch.zeros(1),
        "classifier.head.weight": torch.zeros(1),
    }
    converted = convert_state_dict(original)
    assert "vmamba.embeddings.conv1.weight" in converted
    assert "vmamba.encoder.layers.0.downsample.conv.weight" in converted
    assert "vmamba.layernorm.weight" in converted
    assert "classifier.weight" in converted


def test_no_timm_or_custom_cuda_required():
    import vmamba.modeling_vmamba as modeling

    source = Path(modeling.__file__).read_text()
    assert "import timm" not in source
    assert "import yacs" not in source
    assert "import fvcore" not in source
