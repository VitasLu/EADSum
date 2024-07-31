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
   python run_utils.py --dataset cnndm
   ```

2. Start training:
   ```bash
   python DT/run.py --from_pretrained google-t5/t5-base --dataset cnndm --model_type task_prefix --label_type gt --llm palm --alpha 0.5 --batch_size 4
   ```

3. Standard Finetuning
   ```bash
   python DT/run.py --from_pretrained google-t5/t5-base --dataset cnndm --model_type standard --label_type gt --batch_size 4
   ```

## Inference

To run inference using a trained model:

```bash
python inference.py 
```
