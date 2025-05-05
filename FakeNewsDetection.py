
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import requests
from bs4 import BeautifulSoup
import os

def download_nltk_resources():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        # Additional download for punkt_tab if needed
        try:
            word_tokenize("test")  # This will trigger the punkt_tab load
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")
        return False
    return True

# This will load datasets
fake_path = r"##Path to Your dataset## "
real_path = r"##Path to Your Dataset##"

def load_and_prepare_data(fake_path, real_path):
    try:
        fake_df = pd.read_csv(fake_path)
        real_df = pd.read_csv(real_path)
        
        # Print first few rows to verify data
        print("\nSample fake news headlines:")
        print(fake_df['title'].head(3))
        print("\nSample real news headlines:")
        print(real_df['title'].head(3))
        
        # Add labels
        fake_df['label'] = 'fake'
        real_df['label'] = 'real'
        
        # Combine datasets - using title as primary text feature
        df = pd.concat([fake_df[['title', 'label', 'news_url']], 
                        real_df[['title', 'label', 'news_url']]], 
                       ignore_index=True)
        df = df.rename(columns={'title': 'text'})
        
        print(f"\nDataset loaded with {len(df)} entries ({len(fake_df)} fake, {len(real_df)} real)")
        return df
    
    except FileNotFoundError:
        print(f"\nError: Dataset files not found at:\n{fake_path}\n{real_path}")
        print("Please check the file paths and try again.")
        return None
    except Exception as e:
        print(f"\nError loading data: {str(e)}")
        return None

# It will do Text preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set()

def initialize_text_processor():
    global stop_words
    try:
        stop_words = set(stopwords.words('english'))
    except:
        print("Warning: Could not load stopwords, using empty set")
        stop_words = set()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    try:
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user @ references and '#' from tweet
        text = re.sub(r'\@\w+|\#', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        
        return ' '.join(tokens)
    except Exception as e:
        print(f"Warning: Error preprocessing text: {e}")
        return ""

def train_model(df):
    try:
        # It willPreprocess the text data
        df['processed_text'] = df['text'].apply(preprocess_text)
        
        # Remove empty texts after preprocessing
        df = df[df['processed_text'].str.strip() != '']
        
        if len(df) == 0:
            raise ValueError("No valid text data remaining after preprocessing")
        
        # Split data
        X = df['processed_text']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Vectorize text
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
        tfidf_train = tfidf_vectorizer.fit_transform(X_train)
        tfidf_test = tfidf_vectorizer.transform(X_test)
        
        # Train model
        pac = PassiveAggressiveClassifier(max_iter=50)
        pac.fit(tfidf_train, y_train)
        
        # Evaluate model
        y_pred = pac.predict(tfidf_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel trained with accuracy: {accuracy*100:.2f}%")
        
        return pac, tfidf_vectorizer
    except Exception as e:
        print(f"\nError during model training: {str(e)}")
        raise

def extract_text_from_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text from common article tags
        article_text = ' '.join([p.get_text() for p in soup.find_all(['p', 'article', 'div'])])
        return article_text
    except Exception as e:
        print(f"Error extracting text from URL: {e}")
        return ""

def predict_news(input_text, model, vectorizer):
    # Check if input is a URL
    if input_text.startswith('http://') or input_text.startswith('https://'):
        text = extract_text_from_url(input_text)
        if not text:
            return "Could not extract text from URL. Please try with direct text.", 0
    else:
        text = input_text
    
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    if not processed_text.strip():
        return "No meaningful text could be extracted or processed.", 0
    
    # Vectorize
    text_vector = vectorizer.transform([processed_text])
    
    # Predict
    prediction = model.predict(text_vector)[0]
    decision_score = model.decision_function(text_vector)[0]
    
    # Calculate confidence score (convert to probability-like value)
    if prediction == 'real':
        confidence = (decision_score + 1) / 2
    else:
        confidence = (-decision_score + 1) / 2
    
    confidence_percent = max(0, min(100, confidence * 100))
    
    return prediction, confidence_percent

# Main execution
if __name__ == "__main__":
    print("\nFake News Detection Tool - Initializing...")
    
    # Download NLTK resources
    if not download_nltk_resources():
        print("Could not download required NLTK data. Trying to continue with limited functionality...")
    
    initialize_text_processor()
    
    # Load data
    df = load_and_prepare_data(fake_path, real_path)
    
    if df is not None:
        try:
            # Train model
            model, vectorizer = train_model(df)
            
            # User interface
            print("\nFake News Detection Tool")
            print("-----------------------")
            print("Enter a news headline, article text, or URL to check if it's fake or real")
            print("Type 'exit' to quit\n")
            
            while True:
                user_input = input("Enter news text/headline/URL: ").strip()
                
                if user_input.lower() == 'exit':
                    break
                
                if not user_input:
                    print("Please enter some text, a headline, or a URL")
                    continue
                
                prediction, confidence = predict_news(user_input, model, vectorizer)
                
                if prediction in ['fake', 'real']:
                    print(f"\nResult: This news is predicted to be {prediction} (confidence: {confidence:.1f}%)")
                    if prediction == 'fake':
                        print("Warning: This content appears to be unreliable. Verify with trusted sources.")
                    else:
                        print("This content appears to be reliable, but always verify with multiple sources.")
                else:
                    print(prediction)  # This will show the error message if URL extraction failed
                
                print("\n" + "="*50 + "\n")
            
            print("Thank you for using the Fake News Detection Tool!")
        
        except Exception as e:
            print(f"\nFatal error: {str(e)}")
            print("The tool cannot continue. Please check the error and try again.")
