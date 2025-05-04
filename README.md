# Sentiment-Analysis-with-BER-RAG-based-System
This project uses a RAG-enhanced BERT model for emotion classification on tweets. It outperforms a fine-tuned BERT baseline across accuracy, precision, recall, and F1-score, showing the benefit of contextual retrieval in improving classification performance.


# 🔍 Emotion Detection on Twitter Using Transformer Models and RAG-Based Ensemble Learning


This project implements a Retrieval-Augmented Generation (RAG)-style sentiment classification system on tweets using BERT and a context retriever built with SentenceTransformers.

## 📂 Dataset

- **Source:** [Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)
- **Samples Used:** 100,000 tweets
- **Classes:** Positive (4), Negative (0)

## 🚀 Features

- BERT-based classifier with contextual augmentation
- Uses `SentenceTransformer` to retrieve top-k contextual knowledge
- Fuses tweet with retrieved context before classification
- Supports inference on custom tweets

## 🧠 Model

- Model: `bert-base-uncased` from Hugging Face Transformers
- Context Encoder: `all-MiniLM-L6-v2` from SentenceTransformers
- Fine-tuned for binary sentiment classification

## 🛠️ Installation

```bash
pip install transformers==4.11.3 tokenizers==0.10.3 sentence-transformers faiss-cpu datasets scikit-learn pandas
```

## 🏃‍♂️ How to Run
1.Clone this repo

2. Download the Sentiment140 dataset and place it in the root as training.1600000.processed.noemoticon.csv

3.Run the notebook or script:
  ```bash 
     python Twitter Tweet Recognition with BERT RAG.py
```

## 🧪 Inference Example

predict_sentiment_with_context("I absolutely love this weather!")


## 📚 Contextual Knowledge Base
Used simple hand-crafted definitions to augment tweet understanding:

"Positive sentiment is associated with joy, satisfaction, or approval."

"Negative sentiment involves emotions like anger, sadness, or frustration."






