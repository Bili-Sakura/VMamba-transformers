# VMamba custom pipeline

VMamba is packaged as a **standalone** Hugging Face custom pipeline. It depends on
`transformers` as a library and does **not** require a PR into
`huggingface/transformers`.

## Install

```bash
pip install torch transformers pillow torchvision
pip install -e .
```

## Image classification

```python
from vmamba import pipeline

pipe = pipeline(variant="tiny")              # random init
# pipe = pipeline(model="./vmamba-tiny-s1l8")  # converted official weights
print(pipe("cat.jpg", top_k=5))
```

Tasks:

| `task=` | class | output |
| --- | --- | --- |
| `"image-classification"` (default) | `VMambaImageClassificationPipeline` | `[{label, score}, ...]` |
| `"feature-extraction"` | `VMambaFeatureExtractionPipeline` | pooled embedding |

## Convert official checkpoints

```bash
python -m vmamba.convert_vmamba_original_to_hf \
    --variant tiny \
    --original_checkpoint vssm1_tiny_0230s_ckpt_epoch_264.pth \
    --pytorch_dump_folder_path ./vmamba-tiny-s1l8
```

## Hub + `trust_remote_code`

`vmamba.save_pretrained(path, pipe)` copies `pipeline.py` and the model modules
next to `config.json` and writes `auto_map` / `custom_pipelines`. Push that
folder to the Hub, then:

```python
from transformers import pipeline as hf_pipeline

hf_pipeline("image-classification", model="you/vmamba-tiny", trust_remote_code=True)
```
