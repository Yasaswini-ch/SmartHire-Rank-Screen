import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# Load dataset
df = pd.read_csv(r"C:\RESUME SCREENING AND RANKING\UpdatedResumeDataSet.csv")

# Initialize lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r"http\S+", " ", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z]", " ", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    words = re.findall(r'\b\w+\b', text)  # Tokenize without punkt
    words = [word for word in words if word not in stopwords.words("english")]  # Remove stopwords
    words = [lemmatizer.lemmatize(word) for word in words]  # Apply lemmatization
    words = [stemmer.stem(word) for word in words]  # Apply stemming
    return " ".join(words)

# Apply preprocessing
df["Cleaned_Resume"] = df["Resume"].apply(preprocess_text)

# Convert categories to numbers
category_mapping = {category: idx for idx, category in enumerate(df["Category"].unique())}
df["Category_ID"] = df["Category"].map(category_mapping)

# Train TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df["Cleaned_Resume"]).toarray()
y = df["Category_ID"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save model and vectorizer
pickle.dump(clf, open("model/clf.pkl", "wb"))
pickle.dump(tfidf, open("model/tfidf.pkl", "wb"))

print("Model trained and saved successfully!")
