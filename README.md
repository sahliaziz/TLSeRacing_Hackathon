# TLSeRacing Hackathon — Perception Module

Modality 3: Image classification using transfer learning on CIFAR-100.

---

## Overview

This is my code for the perception modality of [TLS'e Racing](https://tlseracing.fr/)'s driverless division [hackathon](https://github.com/IsmaTIBU/TLSeRacing_Hackathon).
The notebook finetunes a MobileNetV2 model pretrained on ImageNet to classify images from the CIFAR-100 dataset (100 classes).
The approach uses a two-phase training strategy — first training only the classification head, then fine-tuning the upper layers of the backbone — with data augmentation to improve generalization.

---

## How to Run

1. Clone the repository and navigate to the perception folder:
   ```bash
   git clone https://github.com/sahliaziz/TLSeRacing_Hackathon.git
   cd TLSeRacing_Hackathon
   ```

2. Install dependencies with uv:
   ```bash
   uv sync
   ```

3. Register the virtual environment as a Jupyter kernel:
   ```bash
   uv run python -m ipykernel install --user --name tlseracing-perception --display-name "perception"
   ```

4. Launch Jupyter and select the `perception` kernel:
   ```bash
   uv run jupyter notebook perception.ipynb
   ```

A GPU is strongly recommended. The notebook automatically configures GPU memory growth and enables mixed precision if available.

---

## Model

MobileNetV2 (ImageNet weights) with a small classification head added on top: global average pooling, batch normalization, dropout, a 256-unit dense layer, and a final 100-class softmax output.

Training runs in two phases. In phase 1, the backbone is fully frozen and only the head is trained for 15 epochs. In phase 2, the top layers of the backbone are unfrozen and the whole model is fine-tuned for up to 35 more epochs. Early stopping and learning rate reduction are used throughout.

Images are resized from 32×32 to 224×224 and augmented during training with random flips, rotations, zooms, brightness, and contrast adjustments.

---

## Results

Test accuracy on CIFAR-100: **75%**

The final presentation with detailed results and analysis is included in this repository.

---

## Stack

Python 3.13, TensorFlow/Keras, NumPy, Matplotlib