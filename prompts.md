# Business Problem Statement

In today’s social media-driven world, analyzing sentiment on platforms like Twitter is crucial for businesses aiming to understand public opinion and trends. This project seeks to develop an NLP pipeline capable of processing Twitter sentiment data to gain insights into how entities (such as brands, products, or individuals) are perceived. 

The primary objective is to create a DIY solution that includes all foundational NLP processing techniques (stemming, lemmatization, count vectorization, TF-IDF) and advanced embeddings (Word2Vec, GloVe). Additionally, by implementing a custom Encoder-Decoder model in PyTorch, the project will explore sentiment prediction using an unsupervised deep learning approach.

## Key Questions:
1. How can basic NLP transformations help in enhancing sentiment classification accuracy?
2. What role do Word Embeddings play in understanding contextual relationships within tweets?
3. How can a custom-built Encoder-Decoder model contribute to handling nuances in sentiment prediction?

---

# Project Prompts

## 1. Basic Text Processing and Transformation
**Prompt**:  
Implement and compare various basic NLP transformations such as stemming, lemmatization, count vectorization, and TF-IDF on the `tweet_content` field. Analyze how each transformation affects the interpretability and dimensionality of the data.

**Analysis Focus**:  
Observe how each technique impacts sentiment classification and entity recognition accuracy. Discuss the limitations of each transformation for real-time sentiment analysis.

---

## 2. Exploring Word Embeddings
**Prompt**:  
Using the processed text data, apply Word2Vec and GloVe embeddings to generate vectorized representations of each tweet. Compare the embeddings to evaluate which model captures context better, specifically in relation to positive, negative, and neutral sentiments.

**Analysis Focus**:  
Examine which embedding model is more effective for detecting sentiment nuances and entity relevance. Report the embedding model’s impact on the model’s accuracy.

---

## 3. DIY Encoder-Decoder Implementation (PyTorch)
**Prompt**:  
Construct a custom Encoder-Decoder architecture using PyTorch to analyze sentiment. Train this model on your processed Twitter data and evaluate its effectiveness in understanding sentiment shifts within tweets.

**Analysis Focus**:  
Compare the Encoder-Decoder model’s predictions against ground truth sentiment labels. Analyze if the model effectively captures subtle shifts in tone or context within the dataset.

---

## 4. Evaluating Model Performance and Interpretability
**Prompt**:  
Conduct an evaluation comparing traditional NLP techniques and the custom-built Encoder-Decoder model in terms of accuracy, interpretability, and computational efficiency.

**Analysis Focus**:  
Discuss the trade-offs between using basic NLP methods, pre-trained embeddings, and the DIY Encoder-Decoder model for sentiment analysis. Include recommendations on how different techniques may suit various business needs.
