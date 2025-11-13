# BINA Swin Transformer Encoder 
A specialized Swin Transformer–based encoder module — part of the larger BINA research project
---

## Overview

**BINA Swin Encoder** is a customized deep encoder module built upon the [Swin Transformer](https://github.com/microsoft/Swin-Transformer) architecture.  
It is designed for **spatiotemporal data** (multi-frame or video-like input) and optimized to extract hierarchical features at multiple resolutions.

>  **Note:**  
> This repository contains **only one module** out of **five total modules** that make up the complete BINA system.  
> The full project includes additional temporal fusion, decoding, and prediction components that will be released later.

---

## Key Features

- **Spatiotemporal Support:** Handles 5D input tensors `(B, T, C, H, W)` instead of standard 4D images.  
- **Multi-Scale Feature Extraction:** Produces skip connections and a final bottleneck for downstream decoders.  
- **Dynamic Channel Adaptation:** Automatically adjusts pretrained weights for inputs with non-RGB channel counts.  
- **Pretrained Weight Compatibility:** Seamlessly loads official Swin Transformer weights and adapts them.  
- **Memory-Efficient Checkpointing:** Built-in optional gradient checkpointing for large-scale training.  
- **Clean, Google-style Documentation:** Every class and method includes clear and formal docstrings for readability.

---

##  Architecture Overview

The encoder follows the hierarchical Swin Transformer design:
Input (B, T, C, H, W)
↓ Patch Embedding
↓ Stage 1 — BasicLayer (Skip 1)
↓ Stage 2 — BasicLayer (Skip 2)
↓ Stage 3 — BasicLayer (Skip 3)
↓ Stage 4 — Bottleneck Output
- `skip_outputs_flat`: List of feature maps for skip connections  
- `bottleneck_img_flat`: Final downsampled representation used for decoding or prediction  

---

## Research Context

The BINA project is a multi-module system exploring hierarchical attention and feature extraction for temporal–spatial modeling.
This encoder module focuses on feature representation learning, which later connects to transformer-based temporal fusion and decoders.

---

## License

This project is released under the MIT License, following the original Swin Transformer license.
You are free to use, modify, and build upon this code with attribution.

---

## Contact

For research collaboration or technical questions:
Danial Farshbaf
[DanielFrashbaf@gmail.com]
[https://github.com/dpoyyyy]

---

## Designed as a bridge between computer vision and temporal intelligence
