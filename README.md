# Generative-Project-Neural-Networks
This project aims to build an image captioning model. It uses the CLIP model from OpenAI to extract rich visual features (embeddings) from images and a GPT-2 model to generate natural language captions based on those visual features. 
```markdown
# Image Captioning Project: Milestone 1 - Data Preparation

This project aims to build an image captioning model using CLIP for image embeddings and GPT-2 for text generation. This first milestone focuses on setting up the environment, downloading and preprocessing the COCO Captions dataset, generating CLIP image embeddings, and tokenizing captions using a GPT-2 tokenizer.

## Table of Contents
1.  [Setup and Dependencies](#1-setup-and-dependencies)
2.  [Dataset Download and Preprocessing](#2-dataset-download-and-preprocessing)
3.  [CLIP Image Embedding Pipeline](#3-clip-image-embedding-pipeline)
4.  [GPT-2 Caption Tokenization](#4-gpt-2-caption-tokenization)
5.  [Saving Processed Data](#5-saving-processed-data)

## 1. Setup and Dependencies
This section initializes the environment, installs necessary libraries, and sets up PyTorch, Hugging Face Transformers (CLIP, GPT-2), and other supporting libraries. It also defines constants for reproducibility and data handling.

**Key Actions:**
-   Installation of `clip` (from OpenAI's GitHub), `torch`, `torchvision`, `transformers`, `pycocotools`, `pillow`, `tqdm`, `matplotlib`.
-   Mounting Google Drive for data persistence.
-   Verification of PyTorch (CUDA availability), Hugging Face Transformers (GPT-2 tokenizer, CLIP model), and other libraries.
-   Setting `DEVICE` ('cuda' or 'cpu'), `SEED` (42), `SUBSET_SIZE` (5000), `MAX_CAPTION_LEN` (64 tokens).
-   Creating a data directory (`/content/drive/MyDrive/image-captioning/data`).

## 2. Dataset Download and Preprocessing
This part handles downloading the COCO 2017 training images and captions, and then samples a subset of the data. It pairs image paths with their corresponding captions.

**Key Actions:**
-   Downloading `train2017.zip` images and `annotations_trainval2017.zip` to Google Drive.
-   Loading COCO annotations and sampling `SUBSET_SIZE` (5000) unique images.
-   Creating `(image_path, caption)` pairs, taking the first caption for each image.
-   Skipping images with missing files.
-   Displaying a few example image-caption pairs.

## 3. CLIP Image Embedding Pipeline
This section loads a pre-trained CLIP model and uses it to generate 512-dimensional embeddings for all sampled images. These embeddings capture the visual features of the images.

**Key Actions:**
-   Loading `openai/clip-vit-base-patch32` model and processor.
-   Defining an `encode_images` function to process images in batches.
-   Generating and L2-normalizing image embeddings for all `SUBSET_SIZE` images.
-   Performing sanity checks on embedding similarity (image vs. itself, image vs. different image).
-   Visualizing similarity search results: finding and displaying images most similar to a query image based on their CLIP embeddings.

## 4. GPT-2 Caption Tokenization
This step prepares the textual captions for the GPT-2 model by tokenizing them and adding special tokens for sequence handling.

**Key Actions:**
-   Loading a pre-trained GPT-2 tokenizer.
-   Adding custom special tokens: `<|startofcaption|>`, `<|endofcaption|>`, `<|pad|>`.
-   Tokenizing all captions, padding them to `MAX_CAPTION_LEN` (64) and truncating if necessary.
-   Generating `input_ids` and `attention_masks` for each caption.
-   Performing sanity checks on tokenization, including decoding examples and analyzing caption length distribution.

## 5. Saving Processed Data
Finally, all the processed data (image embeddings, tokenized captions, tokenizer, and metadata) are bundled and saved to Google Drive for use in subsequent milestones.

**Key Actions:**
-   Creating a save directory (`/content/drive/MyDrive/image-captioning/processed`).
-   Saving PyTorch tensors (`image_embeddings`, `caption_input_ids`, `caption_attention_masks`, `vocab_size`, `max_caption_len`, `clip_emb_dim`) to `milestone1_tensors.pt`.
-   Saving metadata (including `pairs`, configuration, etc.) to `milestone1_metadata.json`.
-   Saving the configured GPT-2 tokenizer to a directory named `tokenizer`.
-   Verifying the saved data by reloading and checking shapes and decoded captions.

This completes Milestone 1, providing all necessary preprocessed data and models for training an image captioning model.
```
