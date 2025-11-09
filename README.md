# Autoformer from scratch

This repository contains a manual reproduction of the core functions of the Autoformer model, as introduced in the paper "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting" by Wu et al. This work was undertaken in July 2021, after the paper's publication but before the release of the official source code. The purpose of this implementation is to provide an early reference and a deeper understanding of the model's architecture for the community.

## Background

The Autoformer model presents a novel approach to long-term time series forecasting. It enhances the Transformer architecture with a deep decomposition scheme and an innovative auto-correlation mechanism. This design allows the model to progressively break down time series into their trend and seasonal components, leading to more accurate long-term predictions. The auto-correlation mechanism replaces the standard self-attention, offering a more efficient way to discover and utilize period-based dependencies.

## Key Features of this Implementation

This manual reproduction focuses on the core components of the Autoformer architecture:

*   **Decomposition Blocks:** The implementation includes the series decomposition blocks that are central to the Autoformer's progressive decomposition capability.
*   **Auto-Correlation Mechanism:** A key contribution of this work is the implementation of the auto-correlation mechanism, which serves as an alternative to self-attention for discovering dependencies in time series data.
*   **Encoder-Decoder Structure:** The overall encoder-decoder architecture of the Autoformer is replicated to demonstrate how the decomposition and auto-correlation mechanisms are integrated.

## Disclaimer

This is an unofficial implementation created for educational and research purposes. It is based on the descriptions and formulas provided in the original paper. As this was developed prior to the official code release, there may be discrepancies in the implementation details. For the official and most up-to-date implementation, please refer to the authors' official repository: [https://github.com/thuml/Autoformer](https://github.com/thuml/Autoformer).
