# Fine-Tuned BART for Text Summarization

This project fine-tunes the `facebook/bart-base` model for text summarization. The implementation uses the Hugging Face Transformers library and TensorFlow. The model is trained on a dataset provided in CSV format. The provided notebook includes steps for data preprocessing, training, inference, and saving the fine-tuned model.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
  - [1. Data Setup](#1-data-setup)
  - [2. Training the Model](#2-training-the-model)
  - [3. Performing Inference](#3-performing-inference)
  - [4. Saving the Model](#4-saving-the-model)
- [Code Explanation](#code-explanation)
  - [Data Loading and Preprocessing](#data-loading-and-preprocessing)
  - [Tokenization](#tokenization)
  - [Model Training](#model-training)
  - [Evaluation](#evaluation)

## Project Overview

The goal of this project is to fine-tune a pre-trained BART model for summarization. The notebook demonstrates a pipeline that includes:

1.  **Loading Data**: The code reads articles and their corresponding summaries from CSV files.
2.  **Preprocessing**: It tokenizes the text data to prepare it for the BART model.
3.  **Training**: The model is fine-tuned on a training dataset.
4.  **Inference**: The fine-tuned model is used to summarize a new text.
5.  **Saving**: The trained model and tokenizer are saved to a directory.

## Features
- **Model**: Utilizes the `facebook/bart-base` model from Hugging Face.
- **Framework**: Built with TensorFlow and uses `tf.data` for the data pipeline.
- **Pipeline**: The notebook contains an end-to-end implementation from data loading to model saving.
- **Data Source**: The code is set up to read data from CSV files.

## Requirements
To run the code in the notebook, the following libraries are imported:

- `transformers`
- `tensorflow`
- `numpy`

*Note: The final cell attempts to use a `rouge` object, which implies a dependency on a library like `rouge-score`, but the import statement is not included in the notebook.*

## Usage

### 1. Data Setup
The notebook is configured to load data from three specific files: `train.csv`, `validation.csv`, and `test.csv`.

Each CSV file is expected to have a header and the following columns:
- `id`
- `article`
- `highlights`

These files must be in the same directory as the notebook.

### 2. Training the Model
The training process is initiated by running the cells in the notebook. The model training is configured with the following code:

```python
model.compile(optimizer=optimizer)
model.fit(tokenized_train_dataset,validation_data=tokenized_validation_dataset,epochs=3,steps_per_epoch=625,validation_steps=187)
```

### 3. Performing Inference
After training, you can perform text summarization using the fine-tuned model. The notebook demonstrates inference with a sample text:

text = """
A drunk teenage boy had to be rescued by security after jumping into a lions' enclosure at a zoo in western India.
Rahul Kumar, 17, clambered over the enclosure fence at the Kamla Nehru Zoological Park in Ahmedabad, and began running towards the animals,
shouting he would 'kill them'. Mr Kumar explained afterwards that he was drunk and 'thought I’d stand a good chance' against the predators.
"""
inputs = tokenizer([text], max_length=1024, truncation=True, return_tensors='tf')
summary_ids = model.generate(
    inputs['input_ids'],
    num_beams=4,
    min_length=20,
    max_length=100
)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary)

This generates a concise, coherent summary that captures the key ideas of the text.

---

### 4. Saving the Model
Once training and inference are complete, you can save the fine-tuned model and tokenizer for later use:

model.save_pretrained('./fine_tuned_bart_summarizer')
tokenizer.save_pretrained('./fine_tuned_bart_summarizer')

This creates a directory named `fine_tuned_bart_summarizer/` that contains all model weights, configuration files, and the tokenizer vocabulary. You can reload them anytime with:

from transformers import BartTokenizer, TFAutoModelForSeq2SeqLM
model = TFAutoModelForSeq2SeqLM.from_pretrained('./fine_tuned_bart_summarizer')
tokenizer = BartTokenizer.from_pretrained('./fine_tuned_bart_summarizer')

---

## Code Explanation

### Data Loading and Preprocessing
The dataset is loaded from CSV files (`train.csv`, `validation.csv`, and `test.csv`) using TensorFlow’s `tf.data.experimental.CsvDataset`. Each dataset contains three columns:
- `id`
- `article`
- `highlights`

Each record is converted into a dictionary with:
def to_dict(*fields):
    return dict(zip(CSV_COLUMNS, fields))

This structure simplifies referencing fields during tokenization and training.

---

### Tokenization
Tokenization is handled using the `BartTokenizer` from Hugging Face. A custom function encodes both the article and its summary (highlights), padding and truncating sequences to fixed lengths and replacing pad tokens with `-100` to ignore them during loss computation. The function is integrated into a TensorFlow data pipeline via `tf.py_function()`, enabling seamless GPU-based preprocessing. The datasets are then shuffled, batched, and prefetched for optimal training performance.

---

### Model Training
The model used is `facebook/bart-base`, loaded via:
from transformers import TFAutoModelForSeq2SeqLM
model = TFAutoModelForSeq2SeqLM.from_pretrained('facebook/bart-base')

The optimizer is defined as:
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

The model is then compiled and fine-tuned using:
model.compile(optimizer=optimizer)
model.fit(
    tokenized_train_dataset,
    validation_data=tokenized_validation_dataset,
    epochs=3,
    steps_per_epoch=625,
    validation_steps=187
)

This process fine-tunes BART to map full articles to concise summaries using supervised training on the provided dataset.

---

### Evaluation
The notebook references a `rouge` object for evaluation, though it is not explicitly defined. For evaluating generated summaries, you can use the `rouge-score` library:

from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
score = scorer.score(reference_summary, generated_summary)
print(score)

This metric compares the overlap between generated and reference summaries, providing a quantitative measure of summarization quality.

---

## Summary of Workflow

| Step | Description |
|------|--------------|
| **1. Data Loading** | Load CSV files containing article–summary pairs. |
| **2. Preprocessing** | Tokenize text and prepare it for model input. |
| **3. Model Training** | Fine-tune the `facebook/bart-base` model for summarization. |
| **4. Inference** | Generate summaries for unseen text using the fine-tuned model. |
| **5. Saving** | Save the fine-tuned model and tokenizer for reuse. |

---

## Results and Observations
The fine-tuned BART model produces fluent, human-like summaries for structured text such as news articles. The summarization quality depends on factors like dataset diversity, model size, and beam search parameters. While `facebook/bart-base` performs effectively for demonstration purposes, larger variants like `facebook/bart-large` or `google/pegasus-xsum` can yield more accurate and context-aware results. Incorporating evaluation metrics such as ROUGE-L and BLEU can help track summarization performance quantitatively during validation.

