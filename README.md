# Point of Interest (POI) Classification Project

Problem: 
  - Currently, locations may not be properly categorized


Objective:
  - Based on the yelp-dataset, create an RNN model that properly categorizes businesses based on their names


Dataset from: https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset/data?select=yelp_academic_dataset_business.json

Update 5/4:
  From the yelp dataset, I've been able to identify 1311 categories from the overall dataset.
  There are 2118 lines of categories in the overture maps categories

  Next steps:
    - Match categories from Yelp to overture
    - Start the RNN model architecture


# Yelp CNN Multi-Label Classifier

This repository contains code to train and deploy a multi-label CNN classifier on the Yelp dataset, using business names and review snippets as input features. It leverages a DistilBERT transformer for embedding extraction, followed by a CNN + fully-connected head for category prediction.

---

## Table of Contents

* [Features](#features)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Dataset Preparation](#dataset-preparation)
* [Training](#training)
* [Inference](#inference)
* [Streamlit Demo](#streamlit-demo)
* [Configuration](#configuration)
* [Results & Evaluation](#results--evaluation)

---

## Features

* **Multi-Label Classification**: Predict multiple business categories simultaneously.
* **Transformer + CNN**: Use DistilBERT for embeddings, fine-tune last layers, then apply a CNN with multiple filter sizes.
* **Review Integration**: Concatenate each business’s name with the first 200 characters of its earliest review.
* **Class Imbalance Handling**: Compute per-class positive weights for `BCEWithLogitsLoss`.
* **Gradient Clipping**: Prevent exploding gradients.
* **Streamlit Demo**: Interactive web app for on-the-fly predictions.

---

## Prerequisites

* Python 3.8 or higher
* PyTorch >= 1.7
* `transformers` library
* scikit-learn
* Streamlit (for demo)

Install packages via:

```bash
pip install torch transformers scikit-learn streamlit
```

---

## Dataset Preparation

1. **Yelp Business Data**: `yelp_academic_dataset_business.json`
2. **Yelp Review Data**: `yelp_academic_dataset_review.json`

Place both files in your working directory, or provide explicit paths when running scripts.

---

## Training

The training script `CNNTrial.py` loads business + review data, constructs input texts, and trains the model.

```bash
python CNNTrial.py \
  --business_json path/to/yelp_academic_dataset_business.json \
  --reviews_json path/to/yelp_academic_dataset_review.json \
  --sample_size 10000 \
  --max_length 32 \
  --batch_size 32 \
  --epochs 3 \
  --num_filters 100 \
  --filter_sizes 2 3 4 \
  --lr 1e-5
```

**Key arguments:**

* `--sample_size`: Number of business records to sample (default: 10000)
* `--max_length`: Max token length for tokenizer (default: 32)
* `--batch_size`: Training batch size (default: 32)
* `--epochs`: Number of epochs (default: 3)
* `--num_filters`: Number of filters per CNN kernel (default: 100)
* `--filter_sizes`: List of CNN kernel sizes (default: `[2,3,4]`)
* `--lr`: Learning rate for AdamW (default: 1e-5)

After training, two artifacts are saved:

* `yelp_cnn_with_reviews.pth`: Model state dict
* `mlb_classes.json`: List of category labels

---

## Inference

Use `cnn_usage.py` (or `yelp_cnn_inference.py`) for command-line predictions:

```bash
python cnn_usage.py \
  --model_path yelp_cnn_with_reviews.pth \
  --classes_path mlb_classes.json \
  --text "Joe's Pizza and Subs" \
  --top_k 5
```

Returns the top K predicted categories with confidence scores.

---

## Streamlit Demo

An interactive demo is provided in `cnn_streamlit_demo.py`.

```bash
streamlit run cnn_streamlit_demo.py
```

* Configure model path, classes JSON, `max_length`, and `top_k` via the sidebar.
* Enter a business name + snippet to see live predictions.

---

## Configuration

* **Model Freezing**: Only the last two layers of DistilBERT are unfrozen for fine-tuning.
* **Dropout**: 0.7 dropout in the CNN head.
* **Hidden Dimension**: 128-unit hidden layer before final output.
* **Gradient Clipping**: Max norm = 1.0

Feel free to adjust these hyperparameters in the scripts to improve performance.

---

## Results & Evaluation

After training, monitor training and validation loss printed per epoch. You may add:

* Precision/Recall/F1 metrics per category
* ROC-AUC for multi-label classification

Feel free to extend the code for deeper analysis or additional logging (e.g., TensorBoard).

---

**Author**: \[Nataniel Jayaseelan]

**License**: Apache 2.0
