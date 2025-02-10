def load_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    import re
    # Remove null values
    data = data.dropna()
    # Clean text data
    data['news'] = data['news'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x.lower()))
    return data

def feature_engineering(data):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    features = vectorizer.fit_transform(data['news']).toarray()
    return features, vectorizer