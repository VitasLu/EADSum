# EADSum

Brief description of your project.

## Table of Contents
- [Installation](#installation)
- [Training](#training)
- [Inference](#inference)

## Installation

```bash
git clone https://github.com/VitasLu/EADSum.git
cd EADSum
conda env create -f environment.yml
conda activate EADSum
```

## Training

To train the model, follow these steps:

1. Prepare your dataset:
   ```bash
   python prepare_data.py --input_dir /path/to/raw/data --output_dir /path/to/processed/data
   ```

2. Start training:
   ```bash
   python train.py --data_dir /path/to/processed/data --output_dir /path/to/model
   ```

3. Monitor training progress:
   ```bash
   tensorboard --logdir /path/to/model/logs
   ```

## Inference

To run inference using a trained model:

```bash
python inference.py --model_path /path/to/model/checkpoint.pth --input_file /path/to/input/data.txt
```
