# MoE Deepfake Detection

## Overview

This repository contains a research-oriented PyTorch implementation of a Mixture-of-Experts (MoE) deepfake detector inspired by MoE-FFD. The current setup uses a frozen ViT backbone with LoRA-based attention adaptation, adapter experts for local feature refinement, and MoE routing for expert selection.

The project currently focuses on binary face forgery detection (`real` vs `fake`) from cropped face images extracted from video datasets. The primary dataset used so far is Celeb-DF, and the preprocessing pipeline has also been extended to FaceForensics++ subsets (`original`, `Deepfakes`, `Face2Face`, `FaceSwap`) for broader manipulation diversity.

## Current Focus

- Frozen ViT backbone (`vit_base_patch16_224`)
- MoE-LoRA for attention adaptation
- MoE-Adapter for local texture refinement
- Binary classification on cropped face images
- Data-centric experimentation for better generalization across datasets and manipulation types

## Repository Structure

```text
moe-deepfake/
├── data/
│   ├── dataset.py
│   ├── transforms.py
│   ├── video_to_frames.py
│   └── video_to_frames_ffpp.py
├── engine/
│   ├── eval.py
│   ├── loss.py
│   └── train.py
├── models/
│   ├── adapter_experts.py
│   ├── gating.py
│   ├── model.py
│   ├── moe_adapter.py
│   ├── moe_lora.py
│   ├── transformer_block.py
│   └── vit_backbone.py
├── utils/
│   ├── config.py
│   └── metrics.py
├── main.py
├── requirements.txt
└── README.md
```

## Data Pipelines

### Celeb-DF

Celeb-DF preprocessing is handled by:

- `data/video_to_frames.py`

This pipeline supports:

- frame sampling from raw videos
- face cropping with MTCNN
- split-aware processed image export
- manifest generation for training and evaluation

### FaceForensics++ C23

FaceForensics++ preprocessing is handled separately by:

- `data/video_to_frames_ffpp.py`

The current FF++ pipeline only uses:

- `original`
- `Deepfakes`
- `Face2Face`
- `FaceSwap`

and exports cropped faces plus manifests into:

- `data/processed/faceforensicspp_c23`

## Training

The current training entrypoint is:

```bash
python main.py train-celebdf \
  --processed-root data/processed/celebdf \
  --batch-size 8 \
  --epochs 1 \
  --num-workers 0 \
  --image-size 224 \
  --device mps
```

## Research Direction

The current direction is to improve generalization rather than only optimize in-domain performance. In particular, the next stage is expected to explore:

- Celeb-DF as the main realistic benchmark
- FaceForensics++ as a manipulation-diversity source
- SBI-style synthetic forgeries as a generalization regularizer

The model architecture is intended to remain fixed while data composition and training strategy are studied more carefully.

## Notes

- `data/raw/`, `data/processed/`, checkpoints, outputs, and notebooks are ignored by Git.
- The repository tracks code only; datasets and generated artifacts stay local.
- This project is being developed incrementally and is intended for research experimentation rather than polished production use.
