import pandas as pd
import re
import numpy as np
import nltk
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.stem import PorterStemmer, WordNetLemmatizer
# Download NLTK resources if not already installed
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the true news dataset
true_df = pd.read_csv("D:\\naan mudhalvan\\True.csv", encoding='utf-8')
true_df['title'] = 'true'  # Add label 'true' to all rows

# Load the fake news dataset
fake_df = pd.read_csv("D:\\naan mudhalvan\\Fake.csv", encoding='utf-8')
fake_df['title'] = 'fake'  # Add label 'fake' to all rows

# Combine the datasets
combined_df = pd.concat([true_df, fake_df], ignore_index=True)

# Data cleaning
combined_df['text'] = combined_df['text'].apply(lambda x: re.sub('<[^>]+>', '', x))
combined_df['text'] = combined_df['text'].replace('[^\w\s]', '')
combined_df['text'] = combined_df['text'].apply(lambda x: x.lower())
combined_df['text'] = combined_df['text'].apply(lambda x: re.sub('\d+', '', x))

# Tokenization
combined_df['text'] = combined_df['text'].apply(lambda x: word_tokenize(x))

# Stop-word removal
stop_words = set(stopwords.words('english'))
combined_df['text'] = combined_df['text'].apply(lambda x: [word for word in x if word not in stop_words])

# Split the dataset into training and testing sets
X = combined_df['text'].apply(lambda x: ' '.join(x))  # Join tokens into strings
y = combined_df['title']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenization and Padding
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_sequence_length = 250
X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length)

# Encode labels for the deep learning model
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Build and compile a simple LSTM model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_sequence_length))
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Train the deep learning model
model.fit(X_train_pad, y_train_encoded, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the deep learning model
y_pred_dl = model.predict(X_test_pad)
y_pred_dl_binary = (y_pred_dl > 0.5).flatten()

accuracy_dl = accuracy_score(y_test_encoded, y_pred_dl_binary)
precision_dl = precision_score(y_test_encoded, y_pred_dl_binary)
recall_dl = recall_score(y_test_encoded, y_pred_dl_binary)
f1_dl = f1_score(y_test_encoded, y_pred_dl_binary)

print("Deep Learning Model Results:")
print(f'Accuracy: {accuracy_dl:.2f}')
print(f'Precision: {precision_dl:.2f}')
print(f'Recall: {recall_dl:.2f}')
print(f'F1-score: {f1_dl:.2f}')

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
# Preprocess the text data for the Naive Bayes classifier
def preprocess_text(text):
    words = text.split()
    words = [lemmatizer.lemmatize(stemmer.stem(word)) for word in words if word not in stop_words]
    return ' '.join(words)

X_train_nb = X_train.apply(preprocess_text)
X_test_nb = X_test.apply(preprocess_text)

# Vectorize the text data using TF-IDF (Term Frequency-Inverse Document Frequency)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_nb)
X_test_tfidf = tfidf_vectorizer.transform(X_test_nb)

# Train a simple Naive Bayes classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the test data
y_pred_nb = naive_bayes_classifier.predict(X_test_tfidf)

# Evaluate the Naive Bayes model
accuracy_nb = accuracy_score(y_test, y_pred_nb)
confusion_nb = confusion_matrix(y_test, y_pred_nb)
classification_rep_nb = classification_report(y_test, y_pred_nb)

print("Naive Bayes Model Results:")
print(f"Accuracy: {accuracy_nb}")
print("Confusion Matrix:")
print(confusion_nb)
print("Classification Report:")
print(classification_rep_nb)
