# SmartHire-Rank-Screen
Let's check the SmartHire Rank&Screen is an AI-driven tool for automated resume screening and ranking. Using NLP and machine learning, it matches resumes to job descriptions with TF-IDF and predicts job roles via a Random Forest Classifier. Its Streamlit-based interface streamlines recruitment, helping recruiters find top candidates efficiently.

# SmartHire Rank & Screen

## 📌 Overview
**SmartHire Rank & Screen** is an AI-powered resume screening and ranking system designed to streamline the hiring process. By leveraging **Natural Language Processing (NLP) and Machine Learning**, the application analyzes and ranks resumes based on their relevance to a given job description.

💡 **SmartHire Rank & Screen – Smarter Screening, Better Ranking.**

## 🚀 Features
- **Automated Resume Screening**: Extracts and preprocesses text from PDFs, DOCX, and TXT files.
- **Job Matching & Ranking**: Uses **TF-IDF vectorization** and **cosine similarity** to rank resumes based on job descriptions.
- **ML-Based Job Role Prediction**: Trained **Random Forest Classifier** to predict job categories from resumes.
- **Interactive UI**: Built with **Streamlit** for easy usability.
- **Supports Multiple File Formats**: Works with PDF, Word (DOCX), and plain text files.

## 🛠️ Technologies Used
- **Python** (Core language)
- **NLTK** (Text preprocessing)
- **Scikit-learn** (Machine Learning & TF-IDF Vectorization)
- **Random Forest Classifier** (Resume classification)
- **Streamlit** (Web UI)
- **PyPDF2 & python-docx** (Resume text extraction)
- **Pickle** (Model serialization)

## 🔧 Installation & Setup

1. **Clone the repository**:
   ```sh
   git clone https://github.com/yasaswini-ch/smarthire-rank-screen.git
   cd smarthire-rank-screen
   ```

2. **Create a virtual environment** (Recommended):
   ```sh
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App**:
   ```sh
   streamlit run app.py
   ```

## 📂 File Structure
```

├── model/
│   ├── clf.pkl  # Trained ML model
│   ├── tfidf.pkl  # TF-IDF Vectorizer
├── app.py  # Streamlit application
├── resume_data.csv # Resume dataset used for training
├── requirements.txt  # Dependencies
├── README.md  # Documentation
```

## 🎯 Usage
1. **Upload resumes** in PDF, DOCX, or TXT format.
2. **Enter a job description** in the text box.
3. Click **Process**, and the model will:
   - Preprocess and clean resumes.
   - Predict job categories.
   - Rank resumes based on similarity to the job description.
4. **View ranked resumes** in a sorted table.

## 📌 Future Improvements
- **Deep Learning Integration** (BERT for better text understanding)
- **More File Formats Support** (CSV, JSON, etc.)
- **Dashboard Analytics** (More insights into candidate selection)

## 🤝 Contributing
We welcome contributions! Feel free to fork this repo, create a new branch, and submit a pull request.

## 📜 License
This project is licensed under the **MIT License**.

---
💡 **SmartHire Rank & Screen – Smarter Screening, Better Ranking.**


