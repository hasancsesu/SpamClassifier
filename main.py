import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Make sure 'spam.csv' is in the same directory as this Python script,
# or provide the full path to it.
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: spam.csv not found. Make sure it's in the correct directory.")
    exit()

# Display basic information about the dataset
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nDataset Description (Numerical columns):")
print(df.describe())

print("\nValue counts for 'v1' (spam/ham):")
print(df['v1'].value_counts())

# --- Data Preprocessing & Feature Engineering ---

# 1. Rename columns for clarity
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
print("\nColumns renamed successfully:")
print(df.head()) # Displaying first few rows with new column names

# 2. Add a new feature: text_length
df['text_length'] = df['text'].apply(len)
print("\n'text_length' column added:")
print(df.head()) # Displaying first few rows with the new column

# 3. Drop unnecessary columns (if any, based on initial look)
# The dataset has 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4' which are usually empty.
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
print("\nUnnecessary columns dropped:")
print(df.head())

# Display descriptive statistics for the new text_length feature
print("\nDescriptive statistics for 'text_length':")
print(df['text_length'].describe())

# Display value counts for target again to confirm no changes
print("\nUpdated value counts for 'target' (spam/ham):")
print(df['target'].value_counts())

# --- Text Preprocessing (using NLTK) ---

import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize stemmer and stopwords
ps = PorterStemmer()
stopwords_list = stopwords.words('english') # Get English stopwords

def transform_text(text):
    text = text.lower()
    y = []
    for i in text.split():
        if i.isalnum() and i not in stopwords_list:
            y.append(ps.stem(i))

    return " ".join(y)


# Apply the transformation to the 'text' column
df['transformed_text'] = df['text'].apply(transform_text)
print("\n'transformed_text' column created with preprocessed messages:")
print(df[['text', 'transformed_text']].head())

# --- Feature Extraction (Text to Numbers) ---

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TfidfVectorizer
# max_features: Limits the number of features (words) to consider, picking the most frequent ones.
#               This helps manage dimensionality and focus on important words.
#               Adjust this number based on experimentation; 3000 is a common starting point.
tfidf = TfidfVectorizer(max_features=3000)

# Fit and transform the 'transformed_text' column
# .fit_transform() learns the vocabulary from your text and then transforms the text into TF-IDF vectors.
X = tfidf.fit_transform(df['transformed_text']).toarray()

# Display the shape of the resulting feature matrix
print("\nShape of the TF-IDF feature matrix (X):")
print(X.shape) # Should be (number_of_messages, max_features)

# You can also inspect some of the feature names (words) learned by the vectorizer
# print("\nSome feature names (words) learned by TF-IDF:")
# print(tfidf.get_feature_names_out()[:20]) # Shows the first 20 learned words

# --- Prepare Target Variable (y) ---

# Map 'ham' to 0 and 'spam' to 1
# This ensures our target variable is numerical
df['target_numerical'] = df['target'].map({'ham': 0, 'spam': 1})
y = df['target_numerical'].values # Convert the Series to a NumPy array

print("\nShape of the target variable (y):")
print(y.shape)
print("First 5 values of y (numerical labels):")
print(y[:5]) # Displaying first 5 numerical labels


# --- Data Splitting ---

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
# X: our feature matrix (TF-IDF vectors)
# y: our numerical target labels
# test_size=0.20: 20% of the data will be used for testing, 80% for training
# random_state=2: Ensures reproducibility. If you use the same random_state,
#                 you'll get the same split every time you run the code.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=2)

print("\nShapes after train-test split:")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
