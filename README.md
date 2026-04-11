# Generative-Project-Neural-Networks
This project aims to build an image captioning model. It uses the CLIP model from OpenAI to extract rich visual features (embeddings) from images and a GPT-2 model to generate natural language captions based on those visual features. 

Image Captioning Project: Milestone 1 - Data Preparation

This project aims to build an image captioning model using CLIP for image embeddings and GPT-2 for text generation. This first milestone focuses on setting up the environment, downloading and preprocessing the COCO Captions dataset, generating CLIP image embeddings, and tokenizing captions using a GPT-2 tokenizer.

Table of Contents
1.  [Setup and Dependencies](#1-setup-and-dependencies)
2.  [Dataset Download and Preprocessing](#2-dataset-download-and-preprocessing)
3.  [CLIP Image Embedding Pipeline](#3-clip-image-embedding-pipeline)
4.  [GPT-2 Caption Tokenization](#4-gpt-2-caption-tokenization)
5.  [Saving Processed Data](#5-saving-processed-data)

1. Setup and Dependencies
This section initializes the environment, installs necessary libraries, and sets up PyTorch, Hugging Face Transformers (CLIP, GPT-2), and other supporting libraries. It also defines constants for reproducibility and data handling.

**Key Actions:**
-   Installation of `clip` (from OpenAI's GitHub), `torch`, `torchvision`, `transformers`, `pycocotools`, `pillow`, `tqdm`, `matplotlib`.
-   Mounting Google Drive for data persistence.
-   Verification of PyTorch (CUDA availability), Hugging Face Transformers (GPT-2 tokenizer, CLIP model), and other libraries.
-   Setting `DEVICE` ('cuda' or 'cpu'), `SEED` (42), `SUBSET_SIZE` (5000), `MAX_CAPTION_LEN` (64 tokens).
-   Creating a data directory (`/content/drive/MyDrive/image-captioning/data`).

2. Dataset Download and Preprocessing
This part handles downloading the COCO 2017 training images and captions, and then samples a subset of the data. It pairs image paths with their corresponding captions.

**Key Actions:**
-   Downloading `train2017.zip` images and `annotations_trainval2017.zip` to Google Drive.
-   Loading COCO annotations and sampling `SUBSET_SIZE` (5000) unique images.
-   Creating `(image_path, caption)` pairs, taking the first caption for each image.
-   Skipping images with missing files.
-   Displaying a few example image-caption pairs.

3. CLIP Image Embedding Pipeline
This section loads a pre-trained CLIP model and uses it to generate 512-dimensional embeddings for all sampled images. These embeddings capture the visual features of the images.

**Key Actions:**
-   Loading `openai/clip-vit-base-patch32` model and processor.
-   Defining an `encode_images` function to process images in batches.
-   Generating and L2-normalizing image embeddings for all `SUBSET_SIZE` images.
-   Performing sanity checks on embedding similarity (image vs. itself, image vs. different image).
-   Visualizing similarity search results: finding and displaying images most similar to a query image based on their CLIP embeddings.

4. GPT-2 Caption Tokenization
This step prepares the textual captions for the GPT-2 model by tokenizing them and adding special tokens for sequence handling.

**Key Actions:**
-   Loading a pre-trained GPT-2 tokenizer.
-   Adding custom special tokens: `<|startofcaption|>`, `<|endofcaption|>`, `<|pad|>`.
-   Tokenizing all captions, padding them to `MAX_CAPTION_LEN` (64) and truncating if necessary.
-   Generating `input_ids` and `attention_masks` for each caption.
-   Performing sanity checks on tokenization, including decoding examples and analyzing caption length distribution.

5. Saving Processed Data
Finally, all the processed data (image embeddings, tokenized captions, tokenizer, and metadata) are bundled and saved to Google Drive for use in subsequent milestones.

**Key Actions:**
-   Creating a save directory (`/content/drive/MyDrive/image-captioning/processed`).
-   Saving PyTorch tensors (`image_embeddings`, `caption_input_ids`, `caption_attention_masks`, `vocab_size`, `max_caption_len`, `clip_emb_dim`) to `milestone1_tensors.pt`.
-   Saving metadata (including `pairs`, configuration, etc.) to `milestone1_metadata.json`.
-   Saving the configured GPT-2 tokenizer to a directory named `tokenizer`.
-   Verifying the saved data by reloading and checking shapes and decoded captions.

This completes Milestone 1, providing all necessary preprocessed data and models for training an image captioning model.

# Milestone 2: Training CLIP-GPT2 Prefix Model for Image Captioning

This milestone focuses on training a **CLIP-GPT2 Prefix Model** to generate descriptive captions for images. The model acts as a bridge between visual features extracted by CLIP and the language generation capabilities of GPT-2.

## Objectives

*   Load preprocessed image embeddings and tokenized captions from Milestone 1.
*   Define a PyTorch Dataset and DataLoader for efficient training.
*   Implement a `PrefixProjection` layer to connect CLIP embeddings with GPT-2's input space.
*   Construct a `CLIPGpt2CaptionModel` that utilizes precomputed CLIP embeddings and a partially unfrozen GPT-2 (with only the last few layers and LM head trainable).
*   Train the model using AdamW optimizer, a learning rate scheduler with warmup, and mixed precision for efficiency.
*   Monitor training and validation loss, and save the best performing model.
*   Generate and visualize sample captions before and after training to demonstrate learning.

## Model Architecture

The `CLIPGpt2CaptionModel` comprises:

1.  **CLIP Embedding Input**: Uses precomputed 512-dimensional CLIP image embeddings.
2.  **Prefix Projection**: A trainable `PrefixProjection` layer maps the 512-dimensional CLIP embedding into 8 `(num_prefix_tokens)` GPT-2-compatible prefix tokens (each 768-dimensional).
3.  **GPT-2 Language Model**: A `gpt2` model from Hugging Face Transformers, resized to accommodate special tokens (`<|startofcaption|>`, `<|endofcaption|>`, `<|pad|>`). Only the last 2 transformer layers, the LM head, and the token embeddings are unfrozen and trained.

## Training Details

*   **Dataset**: COCO Captions `train2017` subset (5000 images, 2811 valid pairs after filtering).
*   **Splits**: 90% for training, 10% for validation.
*   **Batch Size**: 32
*   **Epochs**: 10
*   **Optimizer**: AdamW with a learning rate of 5e-5 and weight decay of 0.01.
*   **Scheduler**: Linear warmup (100 steps) followed by linear decay.
*   **Mixed Precision**: Enabled for faster training on GPU.
*   **Trainable Parameters**: Approximately 127.6M (out of 215.5M total), focusing on the prefix projection and the unfrozen GPT-2 layers.

## Results

After training, the model successfully learns to generate coherent captions. The training process includes:

*   **Loss Curves**: Plots showing the decrease in both training and validation loss over epochs.
*   **Sample Captions**: Periodic generation of captions for a fixed set of validation images, demonstrating the model's progress in learning to describe images.
*   **Best Model Saving**: The model checkpoint with the lowest validation loss is saved to Google Drive.
*   **Visual Comparison**: A final grid showing input images, ground truth captions, and the captions generated by the trained model.

### Output Files

All generated artifacts are saved to `/content/drive/MyDrive/image-captioning/`:

*   `models/best_model.pt`: The weights of the best performing model.
*   `training_log.json`: Detailed log of training and validation losses, learning rates, and sample captions per epoch.
*   `loss_curves.png`: Plot visualizing the training and validation loss, and learning rate schedule.
*   `final_caption_samples.json`: JSON containing ground truth and generated captions for 10 sample images after training.
*   `trained_caption_samples.png`: A visual grid comparing ground truth and generated captions for the sample images.

# Generative Project — Image Captioning with CLIP + GPT-2

## Overview
An image captioning system that generates natural language descriptions of images by combining **CLIP** (vision encoder) with **GPT-2** (language decoder) using prefix conditioning.

## Architecture

```
Image → CLIP ViT-B/32 → 512-dim embedding → Prefix Projection MLP → 10 prefix tokens (768-dim each) → GPT-2 → Caption
```

- **CLIP (ViT-B/32):** Encodes images into 512-dimensional embeddings. Frozen during training.
- **Prefix Projection MLP:** Learned bridge that converts CLIP embeddings into 10 pseudo-tokens in GPT-2's embedding space. Architecture: `Linear(512→3072) → GELU → Dropout(0.1) → Linear(3072→7680) → LayerNorm`
- **GPT-2:** Language model that generates captions conditioned on visual prefix tokens. Last 4 transformer layers + LM head unfrozen during training.

## Dataset
- **COCO Captions (train2017):** ~20,000 image-caption pairs
- Each image paired with one ground-truth caption
- Captions tokenized with GPT-2 tokenizer + custom special tokens (`<|startofcaption|>`, `<|endofcaption|>`, `<|pad|>`)
- Max caption length: 64 tokens

## Training Details
| Parameter | Value |
|---|---|
| Prefix tokens | 10 |
| GPT-2 unfrozen layers | Last 4 of 12 |
| Batch size | 48 |
| Learning rate | 3e-5 (with linear warmup + decay) |
| Warmup steps | 300 |
| Epochs | 15 |
| Optimizer | AdamW (weight_decay=0.01) |
| Mixed precision | FP16 (AMP) |
| Best validation loss | 1.9547 |
| Training time | ~45 min on T4 GPU |

## Milestone 3: Evaluation & Analysis

### Task 1: Quantitative Metrics
Computed on 500 evaluation images using 3 decoding strategies:

| Strategy | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|---|
| Greedy | 0.6693 | 0.2020 | 0.4043 | 0.4763 | 0.4716 |
| Beam Search (k=5) | 0.3146 | 0.0842 | 0.3829 | 0.3591 | 0.2262 |
| Nucleus (p=0.9) | 0.5885 | 0.1246 | 0.3614 | 0.4168 | 0.3431 |

**Qualitative evaluation:** Model correctly identifies high-level scene content (people, objects, settings) but sometimes misattributes specific activities or objects. This is expected behavior for a prefix-conditioned model trained on 20k images.

### Task 2: Parameter Sensitivity & Fine-Tuning Comparison

**Temperature Sweep:** Evaluated temperatures from 0.3 to 1.5 with nucleus sampling. Lower temperatures produce more deterministic but repetitive captions. Higher temperatures increase diversity but reduce accuracy. Optimal temperature: ~0.5-0.7.

**Beam Width Sweep:** Tested beam widths 1, 3, 5, 8, 10. Performance improves up to beam width 5, with diminishing returns beyond that.

**Fine-Tuning Strategy Comparison:**

| Strategy | Trainable Params | Description |
|---|---|---|
| Prefix Tuning Only | ~4.7M | Only MLP trained, GPT-2 fully frozen |
| Full Fine-Tuning | ~124M | All GPT-2 parameters trainable |
| LoRA | ~1-2M + 4.7M | Low-rank adapters on attention layers |

### Task 3: Visualization
- Bar charts comparing metrics across decoding strategies
- Radar chart showing strategy strengths/weaknesses
- Side-by-side caption comparisons (same image, different strategies)
- Temperature and beam width sensitivity line plots
- Fine-tuning comparison bar charts + training loss curves

## Decoding Strategies

| Strategy | How it works | Pros | Cons |
|---|---|---|---|
| **Greedy** | Always picks highest probability token | Fast, deterministic | Repetitive, gets stuck in loops |
| **Beam Search** | Tracks top-B candidate sequences | Better global coherence | Slower, still deterministic |
| **Nucleus (Top-P)** | Samples from smallest token set with cumulative prob ≥ p | Diverse, adapts to confidence | Occasionally hallucinates |

## Project Structure

```
image-captioning/
├── data/
│   ├── train2017/                    # COCO images
│   └── annotations/
│       └── captions_train2017.json   # COCO captions
├── processed/
│   ├── milestone1_tensors.pt         # CLIP embeddings + tokens (5k)
│   ├── milestone1_metadata.json      # Image paths, captions, config
│   ├── embeddings_20k.pt            # CLIP embeddings (20k)
│   ├── tokens_20k.pt               # Tokenized captions (20k)
│   └── tokenizer/                   # GPT-2 + special tokens
├── models/
│   ├── best_model.pt               # Trained on 5k images (Milestone 2)
│   └── best_model_20k.pt           # Trained on 20k images (improved)
├── Generative project.ipynb         # Main notebook (all milestones)
├── metrics_comparison.png           # Task 1: metric bar chart
├── qualitative_evaluation.png       # Task 1: image + caption grid
├── meteor_distribution.png          # Task 1: per-sample score histogram
├── parameter_sweeps.png             # Task 2: temp + beam plots
├── finetuning_comparison.png        # Task 2: strategy comparison
├── task3_decoding_comparison.png    # Task 3: decoding bar chart
├── task3_radar_chart.png            # Task 3: radar chart
├── task3_caption_comparison.png     # Task 3: side-by-side captions
├── loss_curves.png                  # Milestone 2: training loss
└── loss_curves_20k.png             # 20k model: training loss
```

## How to Run

### Requirements
```
torch
transformers
pycocotools
nltk
rouge-score
pycocoevalcap
peft
matplotlib
pillow
```

### Quick Start — Caption Any Image
1. Open `Generative project.ipynb` in Google Colab
2. Run the **Demo Day** cell (loads model + CLIP automatically)
3. Upload any image when prompted
4. Get captions from 3 decoding strategies (Greedy, Beam Search, Nucleus)

### Retrain the Model
1. Run Cells 1-7 in the training section
2. CLIP encoding: ~10 min (20k images)
3. Training: ~30 min on T4 GPU
4. Model saves to `models/best_model_20k.pt`

## Key Observations

1. **Greedy decoding** achieves highest BLEU-1 (0.6693) due to high unigram precision, but tends to be repetitive.
2. **Beam search** produces more grammatically coherent sentences but scores lower on diversity metrics.
3. **Nucleus sampling** offers the best balance between accuracy and diversity.
4. **Temperature** is the most impactful parameter — optimal range is 0.5-0.7 for this model.
5. **Prefix conditioning** is parameter-efficient (~4.7M trainable params) but has a capacity ceiling due to the single-vector bottleneck from CLIP.
6. Training on **20k images** significantly improved caption relevance compared to the initial 5k subset.

## Limitations
- CLIP compresses images into a single 512-dim vector, losing fine-grained visual details
- Model occasionally confuses similar activities (e.g., "playing frisbee" vs "sitting at a table")
- Limited to COCO-style everyday scenes; struggles with unusual or domain-specific images
- Not comparable to modern vision-language models (BLIP-2, LLaVA) which use billions of parameters

## Tech Stack
- **PyTorch** — deep learning framework
- **Hugging Face Transformers** — CLIP + GPT-2 pretrained models
- **PEFT** — LoRA fine-tuning
- **NLTK / rouge-score / pycocoevalcap** — evaluation metrics
- **Google Colab** — T4 GPU training environment

## Authors
IE7615.38279.202630 — Spring 2026

## References
- [ClipCap: CLIP Prefix for Image Captioning](https://arxiv.org/abs/2111.09734)
- [COCO Captions Dataset](https://cocodataset.org/)
- [OpenAI CLIP](https://openai.com/research/clip)
- [GPT-2](https://openai.com/research/better-language-models)

