import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from docx import Document  # For Word files

# Download NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")

# Load trained model and TF-IDF vectorizer
@st.cache_resource
def load_model():
    clf = pickle.load(open("model/clf.pkl", "rb"))
    tfidf = pickle.load(open("model/tfidf.pkl", "rb"))
    return clf, tfidf

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Preprocess text
def preprocess_text(text):
    text = re.sub(r"http\S+", " ", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z]", " ", text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    words = re.findall(r'\b\w+\b', text)  # Tokenize without 'punkt'
    words = [word for word in words if word not in stopwords.words("english")]  # Remove stopwords
    words = [lemmatizer.lemmatize(word) for word in words]  # Apply lemmatization
    words = [stemmer.stem(word) for word in words]  # Apply stemming
    return " ".join(words)

# Extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() or ""  # Handle cases where extract_text() returns None
    return text.strip()

# Extract text from Word file
def extract_text_from_word(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

# Streamlit App
def main():
    st.title("SmartHire Rank & Screen – Smarter Screening, Better Ranking📄")
    clf, tfidf = load_model()

    job_description = st.text_area("📝 Enter the job description")

    uploaded_files = st.file_uploader(
        "📤 Upload Resumes (PDF, TXT, DOCX)", 
        type=["pdf", "txt", "docx"], 
        accept_multiple_files=True
    )

    if uploaded_files and job_description:
        resumes = []
        filenames = []
        
        for file in uploaded_files:
            if file.type == "application/pdf":
                text = extract_text_from_pdf(file)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = extract_text_from_word(file)
            else:
                text = file.read().decode("utf-8")
            
            cleaned_resume = preprocess_text(text)
            resumes.append(cleaned_resume)
            filenames.append(file.name)

        # Vectorize job description and resumes
        documents = [job_description] + resumes
        vectors = tfidf.transform(documents).toarray()

        # Compute cosine similarity
        job_description_vector = vectors[0]
        resume_vectors = vectors[1:]
        scores = cosine_similarity([job_description_vector], resume_vectors).flatten()

        # Predict job categories
        predictions = clf.predict(resume_vectors)
        category_mapping = {
            15: "Java Developer", 23: "Testing", 8: "DevOps Engineer",
            20: "Python Developer", 24: "Web Designing", 12: "HR",
            13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
            18: "Operations Manager", 6: "Data Science", 22: "Sales",
            16: "Mechanical Engineer", 1: "Arts", 7: "Database",
            11: "Electrical Engineering", 14: "Health and Fitness",
            19: "PMO", 4: "Business Analyst", 9: "DotNet Developer",
            2: "Automation Testing", 17: "Network Security Engineer",
            21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate",
        }
        category_names = [category_mapping.get(pred, "Unknown") for pred in predictions]

        # Create results table
        results = pd.DataFrame({
            "Resume": filenames,
            "Predicted Category": category_names,
            "Score": scores
        })

        # Sort by score
        results = results.sort_values(by="Score", ascending=False)
        st.write(results)

# Run the app
if __name__ == "__main__":
    main()
