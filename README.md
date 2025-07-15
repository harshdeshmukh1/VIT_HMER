# Handwritten Mathematical Expression Recognition

This project uses a pre-trained Vision Transformer (`vit_base_patch16_224`) as an encoder to extract visual features from handwritten mathematical expressions, and a Transformer decoder to generate the corresponding symbol sequence.

## Model Overview

- **Encoder**: Vision Transformer (ViT) from the `timm` library, pretrained on ImageNet.
- **Decoder**: Transformer-based decoder with token and positional embeddings, trained to predict sequences of math symbols.
- **Input**: Images of size `[3, 224, 224]`
- **Output**: Sequences of token IDs representing mathematical expressions.

## Objective

To convert images of handwritten mathematical expressions into tokenized LaTeX-style symbol sequences using an end-to-end deep learning model.

## License

MIT License
