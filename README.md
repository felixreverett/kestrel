# Kestrel

## 1 What is Kestrel?

Kestrel is a lightweight, local language model for Mauritian Creole created as the capstone of my Master's Postgraduate Degree in Computer Science at University College London (UCL). Built on the IBM Granite 4.1 family of open-weight base models, Kestrel aims to democratise artificial intelligence for lower-resource languages - those that have much smaller publicly-available corpora of data than major world languages such as English and French.

The project investigates the fine-tuning of Granite for French- and English-Mauritian Creole translation, using BLEU and chrF++ scores to provide a quantitative analysis of results. 

A full PDF will be publicly available alongside this repository once grading has concluded.

## 2 Model translation scores (chrF++)

**Disambiguation**: Models are divided into base and kestrel variants. Base variants have either 3B or 8B parameters. Base variants with the custom Mauritioan Creole tokeniser exist, but are not meaningfully measurable; they are still zero-shot models. Kestrel version 1 models use the base model's tokeniser, whereas version 2 models use the custom Mauritian Creole tokeniser. All Kestrel models are fine-tuned on the KreolMorisienMT dataset.

### 2.1 English models

| Model           | EN-KM | KM-EN | Avg   |
| :-------------- | ----: | ----: | ----: |
| 4.1-3b-base     | 12.62 | 20.52 | 16.89 |
| kestrel-1-3b-en | 35.97 | 42.73 | 39.51 |
| kestrel-2-3b-en | -     | -     | -     |
| 4.1-8b-base     | 13.97 | 27.86 | 21.10 |
| kestrel-1-8b-en | 37.30 | 41.95 | 39.74 |
| kestrel-2-8b-en | -     | -     | -     |

### 2.2 French models

| Model           | FR-KM | KM-FR | Avg   |
| :-------------- | ----: | ----: | ----: |
| 4.1-3b-base     | 17.00 | 18.78 | 17.94 |
| kestrel-1-3b-fr | 34.67 | 34.24 | 34.46 |
| kestrel-2-3b-fr | -     | -     | -     |
| 8b-base         | 16.91 | 28.30 | 23.04 |
| kestrel-1-8b-fr | 36.28 | 36.99 | 36.70 |
| kestrel-2-8b-fr | -     | -     | -     |

## 3 Understanding this repository

This repository consists of the following folders:

| Folder    | Purpose
| :-------- | :------
| adapters/ | Location of adapter weights after initial training.
| configs/  | Location of config files used to facilitate model training from a single source.
| datasets/ | Local storage of datasets. The KMMT dataset used in this study can be found online.
| evaluation/ | Location for final model results (BLEU and chrF++).
| models/   | Location for models, including SafeTensors and .gguf formats.
| tools/    | The various programs and scripts for the Kestrel pipeline.

To use it yourself, it is recommended to start with the Tools subfolder, which has a dedicated README to describe all the steps of the finetuning pipeline. The Kestrel paper also elaborates on the functionality and usage of each tool.

## 4. Fine-tuning instructions

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
6. (Optional) Create a new base model with custom tokeniser by running the `tk_build-tokeniser.py` script.
7. Run the training script to initialise the base model, apply adapters, and begin QLoRA training loop: `python train.py`
8. Once the training loop has concluded, the resulting LoRA adapters will be saved locally. To finalise the model for offline deployment:
    a. Merge trained adapters back into base IBM Granite model with `python merge.py`
    b. Use llama.cpp to convert merged model into .gguf format.
    c. Serve GGUF model locally via Kestrel CLI.