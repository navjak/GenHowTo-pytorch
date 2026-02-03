# GenHowTo: Learning to Generate Actions and State Transformations from Instructional Videos

[![arXiv](https://img.shields.io/badge/arXiv-2312.07311-b31b1b.svg)](https://arxiv.org/abs/2312.07322)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fwsid3q1VND3nySOTT-3mPRn4F1rw7bt)


This repository provides a **complete, training-ready PyTorch implementation** of the CVPR 2024 paper [**"GenHowTo: Learning to Generate Actions and State Transformations from Instructional Videos"**](https://arxiv.org/abs/2312.07322).

Click here for training and inference run on a T4 GPU: [Colab notebook](https://colab.research.google.com/drive/1fwsid3q1VND3nySOTT-3mPRn4F1rw7bt)

The original authors released inference-only code. This repository includes the missing training code, data loading, and optimization loops required to train the model from scratch.

This implementation has been validated on a sample task (slicing an apple). The model successfully learned to generate the sliced state from a whole apple image given the prompt.

## Features

*   **Training & Inference**: Full pipeline for training, validation, and inference (original release was inference-only).
*   **Optimization**: Supports training on consumer GPUs (e.g., T4) with `accelerate` (fp16) and `bitsandbytes` (8-bit AdamW).
*   **Architecture**: Customizable ControlNet-style encoder with a selectively trained SD 2.1-base decoder.
*   **Zero Convolutions**: Implements zero-initialized convolutions for stable starting gradients.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/navjak/GenHowTo-pytorch.git
    cd GenHowTo-pytorch
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

In the `config.json`, you can adjust parameters such as learning rate, epochs, and the type of model you wish to train (action/final_state model).

### Model Types
The paper describes two types of models:
1.  **Final State Model**: Generates the outcome of an action (e.g., "sliced apple").
2.  **Action Model**: Generates the action itself or an intermediate state.

You can toggle between these two using the `TRAIN_FINAL_STATE_MODEL` flag in `config.json`:

*   `"TRAIN_FINAL_STATE_MODEL": true` - Trains a model to predict the final state.
*   `"TRAIN_FINAL_STATE_MODEL": false` - Trains a model to predict the action.

`config.json`:
```json
{
    "DATA_ROOT": "./data_dir",
    "OUTPUT_DIR": "./checkpoints",
    "LR": 0.00002,
    "EPOCHS": 30,
    "STEPS_PER_LOG": 1,
    "TRAIN_FINAL_STATE_MODEL": true
}
```

## Usage

### 1. Data Preparation
Organize your dataset in the `data_dir` (or your configured `DATA_ROOT`) as follows:

*   `i_init/`: Initial state images (source).
*   `i_state/`: Target final state images (required for Final State Model).
*   `p_state/`: Text prompts for final states (required for Final State Model).
*   `i_action/`: Target action images (required for Action Model).
*   `p_action/`: Text prompts for actions (required for Action Model).

### 2. Training

**Standard Training (CPU/GPU)**
```bash
python train_gpu.py
```

### 3. Inference

```bash
python inference.py --model_path /path/to/weights --img_path /path/to/image.jpeg --prompt "prompt for target state"
```

**Arguments:**
*   `--img_path` (Required): Path to the initial source image.
*   `--prompt` (Required): Text description of the target state/action.
*   `--model_path`: (Required) Path to the trained checkpoint.
*   `--steps`: Number of diffusion steps (default: `50`).
*   `--skip`: Number of timesteps to skip for noisy latent initialization (default: `2`).

## Citation
```bibtex
@inproceedings{soucek2024genhowto,
    title={GenHowTo: Learning to Generate Actions and State Transformations from Instructional Videos},
    author={Sou\v{c}ek, Tom\'{a}\v{s} and Damen, Dima and Wray, Michael and Laptev, Ivan and Sivic, Josef},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2024}
}
```