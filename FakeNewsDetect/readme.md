# Fake News Detection Project
## Project Overview
This project aims to tackle the pervasive issue of fake news by using advanced natural language processing techniques. Leveraged on a pre-trained BERT model from Hugging Face, this application can classify news articles into 'fake' or 'real'. This repository contains all the necessary code and instructions to train and evaluate the model on your dataset.

## Features
- Data Preprocessing: Prepare your text data for training with efficient tokenization and encoding.
- Model Training: Fine-tune a pre-trained BERT model on your labeled news data.
- Evaluation: Assess the model's performance on a validation set.
- Prediction: Use the trained model to predict the veracity of new news articles


## Training the Model

Customize the training by modifying the arguments:

- model: Pre-trained BERT model (default: bert-base-uncased).
- epoch: Number of training epochs (default: 2).
- lr: Learning rate (default: 2e-5).
- bs: Batch size (default: 32).

```python main.py --model bert-base-uncased --epoch 3 --lr 5e-5 --bs 16```
