Fine-Tuned BART for Text Summarization
This project fine-tunes the facebook/bart-base model for abstractive text summarization using TensorFlow and the Hugging Face Transformers library. The model is trained on a custom dataset provided in CSV format and includes scripts for data preprocessing, training, inference, and saving the final model.

Table of Contents
Project Overview

Features

Requirements

Usage

1. Data Setup

2. Training the Model

3. Performing Inference

4. Saving and Loading the Model

Code Explanation

Data Loading and Preprocessing

Tokenization

Model Training

Evaluation

Project Overview
The goal of this project is to create a custom text summarizer by fine-tuning a pre-trained BART model. The notebook handles the entire pipeline:

Loading Data: Reads articles and their corresponding summaries (highlights) from CSV files.

Preprocessing: Tokenizes the text data to make it suitable for the BART model.

Training: Fine-tunes the model on the training dataset and validates it.

Inference: Uses the fine-tuned model to summarize new, unseen text.

Saving: Saves the trained model and tokenizer for future use.

Features
Model: Utilizes the powerful facebook/bart-base model from Hugging Face.

Framework: Built with TensorFlow 2.x and integrates with tf.data for efficient data handling.

Pipeline: End-to-end implementation from data loading to model deployment.

Customizable: Easily adaptable to any article/summary dataset in CSV format.

Requirements
To run this project, you need to install the following libraries. You can install them using pip:

Bash

pip install tensorflow transformers numpy rouge-score
Note: The rouge-score library is needed for the final evaluation cell in the notebook.

Usage
1. Data Setup
Before running the notebook, ensure your data is structured correctly. You will need three files: train.csv, validation.csv, and test.csv.

Each CSV file must contain the following columns:

id: A unique identifier for each entry.

article: The full text that you want to summarize.

highlights: The reference summary for the corresponding article.

Place these files in the same directory as the my_text_summarizer.ipynb notebook.

2. Training the Model
Open the my_text_summarizer.ipynb notebook and run the cells sequentially. The training process is initiated in the following cell:

Python

model.compile(optimizer=optimizer)
model.fit(
    tokenized_train_dataset,
    validation_data=tokenized_validation_dataset,
    epochs=3,
    steps_per_epoch=625,
    validation_steps=187
)
This will train the model for 3 epochs and save the training progress.

3. Performing Inference
To summarize a new piece of text, you can use the trained model object. The notebook provides an example:

Python

text = """
A drunk teenage boy had to be rescued by security after jumping into a lions' enclosure at a zoo in western India.
Rahul Kumar, 17, clambered over the enclosure fence at the Kamla Nehru Zoological Park in Ahmedabad, and began running towards the animals,
shouting he would 'kill them'. Mr Kumar explained afterwards that he was drunk and 'thought I'd stand a good chance' against the predators.
...
"""

# Tokenize the input text
inputs = tokenizer(
    text,
    padding='max_length',
    max_length=512,
    truncation=True,
    return_tensors="tf"
)

# Generate the summary
summary_ids = model.generate(
    inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_length=128
)

# Decode and print the summary
prediction = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(prediction)
4. Saving and Loading the Model
The notebook saves the fine-tuned model and its tokenizer to a directory named myFineTunedBart.

Saving:

Python

model.save_pretrained('myFineTunedBart')
tokenizer.save_pretrained('myFineTunedBart')
Loading for Future Use:
You can easily load the model and tokenizer back in another script without retraining:

Python

from transformers import TFAutoModelForSeq2SeqLM, BartTokenizer

# Load the tokenizer and model from the saved directory
tokenizer = BartTokenizer.from_pretrained('myFineTunedBart')
model = TFAutoModelForSeq2SeqLM.from_pretrained('myFineTunedBart')

# Now you can use them for inference as shown above
Code Explanation
Data Loading and Preprocessing
The data is loaded from CSVs using tf.data.experimental.CsvDataset, which is highly efficient for large datasets. Each row is then converted into a dictionary mapping column names (article, highlights) to their tensor values.

Tokenization
A custom tokenization function tokenize_bart is defined to process the articles and highlights.

Input (article): Truncated or padded to a max_length of 512 tokens.

Target (highlights): Truncated or padded to a max_length of 128 tokens.

Labels: Padding tokens in the target summary are replaced with -100 to ensure they are ignored during the loss calculation.

The function is wrapped in tf.py_function to be seamlessly integrated into the tf.data pipeline.

Model Training
The pre-trained TFAutoModelForSeq2SeqLM is loaded from facebook/bart-base.

An Adam optimizer with a learning rate of 5×10 
−5
  is used.

The model is compiled and trained using the .fit() method on the prepared tf.data.Dataset.

Evaluation
The final cell in the notebook is intended for evaluating the generated summary against a reference using the ROUGE metric. To make it functional, you need to import the rouge library and define the variables.

Here is a corrected and complete example of how to perform the evaluation:

Python

from rouge import Rouge

# This should be the original, full text
# Note: In the notebook, this variable was named 'example_text' but not defined.
# For a real ROUGE score, you should use the reference summary, not the original article.
candidate_summary = prediction # The summary generated by the model

reference_summary = "A 17-year-old boy was rescued after jumping into a lion enclosure at a zoo in Ahmedabad, India. Rahul Kumar was drunk when he ran towards the animals shouting he would kill them. He was caught by guards after falling into a moat." # A human-written reference summary

# Initialize the ROUGE scorer
rouge = Rouge()

# Get the scores
scores = rouge.get_scores(candidate_summary, reference_summary)
print(scores)

# Output might look like:
# [{'rouge-1': {'r': ..., 'p': ..., 'f': ...}, 'rouge-2': ..., 'rouge-l': ...}]
