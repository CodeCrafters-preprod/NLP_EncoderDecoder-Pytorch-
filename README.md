# NLP_EncoderDecoder-Pytorch-
This project implements a sequence-to-sequence (Seq2Seq) model using an Encoder-Decoder architecture to perform sentiment analysis on a Twitter sentiment dataset. The model is built in PyTorch, utilizing GRU (Gated Recurrent Unit) layers to handle text input. The goal of this project is to classify tweets into different sentiment categories (positive, negative, neutral, and irrelevant).

# Table of Contents

Project Overview
Key Concepts
Data Preparation
Training and Evaluation
Getting Started 
Model Architecture
Requirements

# Project Overview

The Encoder-Decoder model is commonly used in natural language processing tasks like machine translation and text generation. Here, we adapt it to perform a classification task—predicting the sentiment of a tweet. This README includes the necessary background, setup, and detailed descriptions of key components.

# Key Concepts

1. Tokenization
Tokenization is the process of splitting text into individual words or tokens. Tokenization allows us to represent each word as an input to the model. Here, we use the "basic_english" tokenizer from PyTorch.

2. Vocabulary
The vocabulary is the set of unique tokens or words that the model will recognize. We build our vocabulary using the tokenized dataset, and it includes special tokens for unknown words ("<unk>") and padding ("<pad>") to handle varying sentence lengths.

3. Embeddings
Word embeddings are dense vector representations of words, mapping words into a continuous vector space where similar words have similar embeddings. This project uses embeddings in both the Encoder and Decoder models to represent input text in a form that the model can process effectively.

4. Encoder
The Encoder processes the input text sequence and generates a context or hidden state that represents the entire input. This hidden state is then passed to the Decoder. In this project, the Encoder uses:

Embedding layer: Converts each word into a dense vector.
GRU layer: Processes the sequence of embeddings to generate the hidden state.
5. Decoder
The Decoder generates the final output (sentiment classification) based on the hidden state provided by the Encoder. The Decoder also has:

Embedding layer: Transforms input into dense vectors.
GRU layer: Processes the sequence to produce hidden states.
Fully connected (Linear) layer: Maps the hidden state to the output classes.
6. Loss Function
We use Cross-Entropy Loss for classification tasks, which calculates the difference between the predicted class probabilities and the actual labels.

7. Backpropagation
Backpropagation is the process of adjusting model weights based on the error between predicted and actual values. Gradients are calculated using the chain rule and are used to update weights to minimize the loss function.

8. Optimization
The Adam optimizer is used to update model parameters. It combines the advantages of both momentum and adaptive learning rates for efficient training.

# Data Preparation

Load Dataset: The Twitter sentiment dataset is loaded and preprocessed.
Preprocessing:
Remove stopwords, punctuation, and lowercase all text.
Perform tokenization, stemming, and lemmatization.
Encoding Sentiment Labels: Encode the sentiment labels into integers:
Positive = 1
Negative = 0
Neutral = 2
Irrelevant = 3
Vectorization: TF-IDF and Word2Vec embeddings are used to represent the text for training.

# Training and Evaluation

Training Loop:
Define the number of epochs and set the model to training mode.
Use Cross-Entropy Loss and the Adam optimizer.
For each batch, calculate loss, perform backpropagation, and update weights.
Evaluation:
The evaluation function calculates the model’s accuracy and classification report on the validation and test sets.
python

# Getting Started

To get started with this project:

Load your Twitter dataset (twitter_data.csv), which should contain columns such as tweet_content (for the text) and sentiment (for the sentiment labels).
Preprocess text: Run preprocess_text to tokenize, remove stopwords, and apply stemming and lemmatization. This function prepares the data for vectorization.
Vectorize text: Use either CountVectorizer or TfidfVectorizer to convert the processed text into numerical representations for the model.
Generate Embeddings: Use Word2Vec for custom embeddings or load pre-trained GloVe embeddings for better word representations.
Train the Model: Train a Logistic Regression model or the Encoder-Decoder model for classification and test its performance.

# Model Architecture

Encoder
The Encoder class initializes an embedding layer and a GRU layer to process the input sequence. It outputs a hidden state that summarizes the input sequence, which the Decoder will use.

Decoder
The Decoder class takes the hidden state from the Encoder and generates a prediction for each time step. It uses an embedding layer, a GRU layer, and a linear layer to produce the final output.

Seq2Seq Model
The Seq2Seq class combines the Encoder and Decoder. It initializes both components and passes the hidden state from the Encoder to the Decoder.

# Requirements

Python 3.7+
PyTorch
NLTK
Scikit-learn
Gensim

# Results

The Encoder-Decoder model, trained on the Twitter sentiment dataset, demonstrates strong performance across both the validation and test sets, indicating effective generalization and accurate sentiment classification.

Validation Accuracy: 90.67%
Test Accuracy: 93.42%

# Contributor	              Role/Responsibilities

## Contributors

| Contributor      | Role/Responsibilities                                       |
|------------------|-------------------------------------------------------------|
| **Khushboo Mittal**       | Data preprocessing, Tokenization |
| **Prachi Tavse**                |Encoder (Embedding, GRU for Hidden State Generation) |
| **Harshita Jangde**                |Decoder (Embedding, GRU, FC Output) and Seq2Seq (Encoder-Decoder Integration) |
| **Tanisha Priya**                | Initialize Model, Train, and Evaluate Accuracy with Classification Report |
