# Dragon Trainer Lab Series
How to Customize a Small Language Model (SLM) for Your Domain

This repository contains a 7-part lab series that walks you through fine-tuning a Small Language Model (SLM) with LoRA + Unsloth on the ncbi/Open-Patients dataset.  
The theme follows our blog series "How to Train Your Dragon", where each lab builds on the last — from environment setup to packaging your trained model for release.

## Lab Overview

- Lab 1 – Colab Environment Setup  
  Prepare a stable Colab environment: GPU check, PyTorch install (auto-fallback), Unsloth and dependencies, test model load and inference.

- Lab 2 – Base Model Loading and Inference  
  Load a base SLM (Mistral or Llama-2), run prompts, explore generation parameters.

- Lab 3 – Data Loading and Tokenization  
  Load the ncbi/Open-Patients dataset, clean, split, and tokenize for causal LM training.

- Lab 4 – LoRA Fine-Tuning  
  Attach LoRA adapters with Unsloth, run domain-adaptive continued pretraining.

- Lab 5 – Hyperparameter Tuning and Optimization  
  Sweep learning rates, LoRA ranks, sequence lengths for quality vs. speed trade-offs.

- Lab 6 – Evaluation and Comparison  
  Compare base vs. tuned model outputs, run perplexity evaluation, produce a qualitative report.

- Lab 7 – Reproducible Packaging and Release  
  Save LoRA adapters, create a model card, push to Hugging Face Hub.

## Dataset: ncbi/Open-Patients

- Source: Hugging Face Dataset Card  
- Description: 180k+ patient descriptions (medical domain)  
- Key Field: description (free-text clinical notes)  
- License: CC BY-SA 4.0 – derivatives must be shared under the same license  
- Use Case in Labs: Used for continued pretraining to adapt the base model to the medical domain

Disclaimer: Models trained on this dataset are for research and educational purposes only and not for clinical use.

## Prerequisites

### Google Colab
- GPU runtime enabled (T4 or better recommended)  
- Hugging Face account (for gated models like Llama-2)  
- Hugging Face CLI token if needed:  
  from huggingface_hub import login  
  login()

### Local (Optional)
- Python 3.10+  
- Conda environment  
- CUDA 11.8+ or 12.x with NVIDIA drivers  
- See lab1_colab_setup.ipynb for detailed install steps

## Repo Structure

- labs/  
  - lab1_colab_setup.ipynb  
  - lab2_inference.ipynb  
  - lab3_data_prep.ipynb  
  - lab4_finetune_lora.ipynb  
  - lab5_tuning.ipynb  
  - lab6_eval.ipynb  
  - lab7_release.ipynb  

- data/  
  - prepare_open_patients.py  

- train/  
  - run.py  
  - config.json  

- eval/  
  - prompts_med.jsonl  

- utils/  
  - env_check.py  
  - seed.py  

- README.md  
- MODEL_CARD.md  
- requirements.txt  

## Quick Start (Colab)

1. Open Lab 1 in Colab  
   - Enable GPU: Runtime → Change runtime type → Hardware accelerator: GPU  
   - Run all cells to install dependencies and test model load.

2. Proceed to Lab 2 to load a base model and test inference.

3. Follow Labs 3–7 in sequence to prepare data, fine-tune, evaluate, and package your model.

## Blog Series Reference

This repo complements the blog series:  
- Lab 1 Blog Post – Colab Environment Setup  
- Labs 2–7 posts will be published weekly

## Disclaimer

This project is for educational purposes only.  
The ncbi/Open-Patients dataset contains medical descriptions and is licensed under CC BY-SA 4.0.  
Any derivative model must comply with the share-alike license and must not be used for clinical decision-making.
