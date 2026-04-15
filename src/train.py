import os
import string
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# Download necessary NLTK corpora explicitly just in case
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Import the data loader module
from data_loader import load_data

def preprocess_text(text: str) -> str:
    """
    Preprocess text according to the standard NLP pipeline:
    1. Lowercasing
    2. Removing punctuation
    3. Tokenization (splitting into words)
    4. Removing stopwords
    """
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Removing punctuation
    # string.punctuation contains '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    text = "".join([char for char in text if char not in string.punctuation])
    
    # 3. Tokenization & 4. Removing stopwords
    stop_words = set(stopwords.words('english'))
    # Tokenize manually by splitting (alternatively we could use nltk.word_tokenize)
    words = text.split()
    
    filtered_words = [word for word in words if word not in stop_words]
    
    return " ".join(filtered_words)

def train_and_evaluate():
    """
    Main function to load data, preprocess, train models, and print evaluations.
    """
    print("=== Step 1: Loading Dataset ===")
    # Assumes data folder is up one level from src, or in current working dir
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    df = load_data(data_dir=data_dir)
    
    print("\n=== Step 2: Data Preprocessing ===")
    df['clean_message'] = df['message'].apply(preprocess_text)
    
    # Map labels to binary values. Spam = 1, Ham (Legitimate) = 0
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    
    print("\n=== Step 3: Train/Test Split ===")
    # Split the dataset: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_message'], 
        df['label_num'], 
        test_size=0.2, 
        random_state=42 # Random state ensures reproducible splits
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    print("\n=== Step 4: Feature Extraction (TF-IDF) ===")
    # Convert words to numerical features using Term Frequency-Inverse Document Frequency
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("\n=== Step 5.1: Training Primary Model (Multinomial Naive Bayes) ===")
    # Naive Bayes works extremely well with text classification tasks like spam filtering
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    
    y_pred_nb = nb_model.predict(X_test_tfidf)
    
    print("--- Naive Bayes Evaluation ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred_nb):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_nb):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_nb):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_nb))
    
    print("\n=== Step 5.2: Training Comparison Model (Logistic Regression) ===")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)
    
    y_pred_lr = lr_model.predict(X_test_tfidf)
    
    print("--- Logistic Regression Evaluation ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred_lr):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_lr):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_lr))
    
    print("\n=== Step 6: Saving Models ===")
    # Save the vectorizer and best performing model (Naive Bayes usually preferred for speed/simplicity here)
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    joblib.dump(nb_model, os.path.join(models_dir, 'naive_bayes_model.pkl'))
    joblib.dump(vectorizer, os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
    
    print(f"Model and Vectorizer saved to {models_dir}")

if __name__ == "__main__":
    train_and_evaluate()
