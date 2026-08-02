# Kestrel

## What is Kestrel?

Kestrel is a lightweight, local language model for Mauritian Creole created as my thesis project for my MSc Computer Science at University College London. Built on IBM Granite 3.0 open-source language model infrastructure, the project aims to democratise AI for lower-resource languages - those that have much smaller publicly-available corpora of data than major world languages such as English and French.

## Technologies and Tools

### Base Model: IBM Granite 3.0

### Environment: Python 3.11 via Anaconda

### Core ML Framework: PyTorch

Fine-t

## Fine-tuning instructions
The training pipeline uses Parameter-Efficient Fine-Tuning (PEFT) and 4-bit quantisation to enable the model to train on consumer-grade GPUs (RTX 4060 8GB, RTX 4070 12GB). To set up the environment, follow the instructions below. Ensure you have Anaconda installed.

1. Create a new environment with Anaconda: `conda create -n kestrel python=3.11`
2. Activate it with `conda activate kestrel`
3. Install Pytorch (for NVIDIA CUDA): `pip install torch torchvision torchaudio --index-url https://download.pytorch.com/whl/cu121`
4. Install core ML stack `pip install transformers datasets peft trl bitsandbytes accelerate`
    a. `transformers`: Core Hugging Face library for loading and interacting with IBM Granite.
    b. `datasets`: Library to handle efficient loading, mapping, and batching of JSONL training data.
    c. `peft`: Injects Low-Rank Adapters (LoRA) into the model, freezing base weights and reducing trainable parameters.
    d. `trl`: Provides the Supervised Fine-Tuning used to calculate training loss.
    e. `bitsandbytes`: Shrinks base model into 4-bit precision (Normal Float 4) as specified in Dettmers et al. (2023).
    f. `accelerate`: PyTorch wrapper to optimise memory allocation and device mamping across GPU.
5. Ensure translation dataset (e.g., KreolMorisienMT) has been downloaded and correctly formatted into .jsonl by the Golang parser in the tools/ directory.
6. Run the training script to initialise the base model, apply adapters, and begin QLoRA training loop: `python train.py`
7. Once the training loop has concluded, the resulting LoRA adapters will be saved locally. To finalise the model for offline deployment:
    a. Merge trained adapters back into base IBM Granite model with `python merge.py`
    b. Use llama.cpp to convert merged model into .gguf format.
    c. Serve GGUF model locally via Kestrel CLI.