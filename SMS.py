# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(file_path):
    df = pd.read_csv('C:\\Users\\AMANDEEP SINGH\\Desktop\\archive\\spam.csv', encoding='latin-1')
    df = df.rename(columns={'v1': 'Label', 'v2': 'Message'})
    df = df[['Label', 'Message']]  # Drop unnecessary columns
    df['Label'] = df['Label'].map({'ham': 0, 'spam': 1})
    return df

def perform_eda(df):
    print("Dataset Overview:")
    print(df.head())
    print("\nClass Distribution:")
    print(df['Label'].value_counts(normalize=True))

def train_model(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f'\nAccuracy: {accuracy:.4f}')
    print('\nConfusion Matrix:')
    print(conf_matrix)
    print('\nClassification Report:')
    print(class_report)

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def predict_messages(model, vectorizer, messages):
    transformed_messages = vectorizer.transform(messages)
    predictions = model.predict(transformed_messages)
    return ["Spam" if pred == 1 else "Ham" for pred in predictions]

# Main script
if __name__ == "__main__":
    # File path to the dataset
    file_path = 'C:\\Users\\AMANDEEP SINGH\\Desktop\\archive\\spam.csv'
    
    # Load and preprocess data
    df = load_data(file_path)
    
    # Perform basic EDA
    perform_eda(df)
    
    # Feature Extraction
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Message'])
    y = df['Label']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    # Save the model and vectorizer
    save_model(model, 'sms_spam_model.pkl')
    save_model(vectorizer, 'vectorizer.pkl')
    
    # Predict new messages
    new_messages = [
        "Free entry in 2 a weekly competition to win", 
        "Hey, are we still meeting for dinner?"
    ]
    predictions = predict_messages(model, vectorizer, new_messages)
    
    # Display predictions for new messages
    print('\nPredictions for new messages:')
    for message, prediction in zip(new_messages, predictions):
        print(f'"{message}" -> {prediction}')
