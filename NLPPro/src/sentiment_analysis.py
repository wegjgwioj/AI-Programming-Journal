from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def analyze_sentiment(texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=1)
    return predictions.numpy()

def extract_features(texts):
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_features = tfidf_vectorizer.fit_transform(texts).toarray()
    return tfidf_features