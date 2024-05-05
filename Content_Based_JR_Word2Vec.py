import os
import streamlit as st
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import matplotlib.pyplot as plt
import seaborn as sns
import PyPDF2
import docx
import re
import requests
import zipfile
import io
import csv


# Function to download the zip file and extract the CSV
def process_github_zip(url, csv_file_name):
    response = requests.get(url)
    if response.status_code == 200:
        # Read the content of the zip file
        zip_data = io.BytesIO(response.content)
        
        # Open the zip file
        with zipfile.ZipFile(zip_data, 'r') as zip_ref:
            if csv_file_name in zip_ref.namelist():
                # Open the CSV file directly from the zip file
                with zip_ref.open(csv_file_name) as csv_file:
                    # Read the CSV file using pandas
                    data = pd.read_csv(csv_file)
                    return data

# URL of the zip file on GitHub
github_zip_url = 'https://github.com/Deke8521/JR-System/raw/main/dice_com-job_us_sample.zip'

# Function to fetch text from Word document
def fetch_text_from_word_doc(file):
    try:
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error("Error extracting text from Word document: " + str(e))
        return None

# Function to fetch text from PDF file
def fetch_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text
    except Exception as e:
        st.error("Error extracting text from PDF: " + str(e))
        return None
    
# Function to clean text
def clean_text(text):
    # Remove special characters and punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Initialise lemmatizer
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into a single string
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Function to load job dataset
@st.cache_data(persist=True)
def load_job_dataset():
    try:
        # Load job data from GitHub
        job_data = process_github_zip(github_zip_url, 'dice_com-job_us_sample.csv')
        
        job_dataset = job_data[['company', 'jobid', 'jobtitle','jobdescription', 'skills']].copy()
        # Check for null values in specific columns
        null_values = job_dataset[['company', 'jobid', 'jobtitle', 'jobdescription', 'skills']].isnull().sum()
        print("Null values in each column:")
        print(null_values)
        # Remove NA values from job profile
        job_dataset.dropna(inplace=True)
        # Combine job skills
        job_dataset['Skills'] = job_dataset['skills'] + ';' + job_dataset['jobdescription']  # Combine 'skills' and 'jobdescription' columns
        # Drop redundant columns
        job_dataset.drop(columns=['skills', 'jobdescription'], inplace=True)
        # Apply clean_text function to the entire job_dataset
        job_dataset = job_dataset.map(clean_text)
        return job_dataset 
    except Exception as e:
        st.error("Error loading job dataset: " + str(e))
        return None

# Main function
def main():
    st.image("https://raw.githubusercontent.com/Deke8521/JR-System/main/Screenshot%202024-04-28%20205652.png", use_column_width=True, width=100)
    st.title("Job Recommendation App")
    job_dataset = load_job_dataset()

    # Initialize resume_text variable
    resume_text = ""

    # Get input method for resume
    resume_input_method = st.selectbox("Select method to input resume:", ("Upload PDF", "Upload Word Document"))
    
    if resume_input_method == "Upload PDF":
        resume_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
        if resume_file is not None:
            with st.spinner("Processing..."):
                resume_text = fetch_text_from_pdf(resume_file)
                if resume_text:
                    # Clean the resume text
                    resume_text = clean_text(resume_text)
                else:
                    st.error("Error processing resume. Please try again.")
    elif resume_input_method == "Upload Word Document":
        resume_file = st.file_uploader("Upload your resume (Word Document)", type=["docx"])
        if resume_file is not None:
            with st.spinner("Processing..."):
                resume_text = fetch_text_from_word_doc(resume_file)
                if resume_text:
                    # Clean the resume text
                    resume_text = clean_text(resume_text)
                else:
                    st.error("Error processing resume. Please try again.")
    if resume_text:
        proceed_button = st.button("Enter")
        if proceed_button:
            with st.spinner("Finding recommended jobs..."):
    
                # Train Word2Vec model on job descriptions
                job_descriptions = [word_tokenize(text) for text in job_dataset['Skills']]
                word2vec_model = Word2Vec(sentences=job_descriptions, vector_size=100, window=5, min_count=1, workers=4)
                
                # Vectorize user's resume
                user_resume_tokens = word_tokenize(resume_text)
                user_resume_embedding = [word2vec_model.wv[token] for token in user_resume_tokens if token in word2vec_model.wv]
                if user_resume_embedding:
                    user_resume_embedding = sum(user_resume_embedding) / len(user_resume_embedding)

                    # Calculate similarity scores between user's resume and job descriptions
                    similarities = []
                    for job_description in job_descriptions:
                        job_description_embedding = [word2vec_model.wv[token] for token in job_description if token in word2vec_model.wv]
                        if job_description_embedding:
                            job_description_embedding = sum(job_description_embedding) / len(job_description_embedding)
                            similarity = cosine_similarity([user_resume_embedding], [job_description_embedding])[0][0]
                            similarities.append(similarity)
                        else:
                            similarities.append(0)

                    # Add similarity scores to job dataset
                    job_dataset['similarity_score'] = similarities

                    # Recommend top-N jobs
                    top_n_jobs = job_dataset.sort_values(by='similarity_score', ascending=False).head(10)
                    
                    average_similarity_score = top_n_jobs['similarity_score'].mean()
                    st.write(f"Average Similarity Score: {average_similarity_score :.4f}")

                    # Display recommended jobs in a table
                    st.subheader("Top 10 Recommended Jobs Using Word2Vec:")
                    table_data = [{"Job Title": job, "Company": company, "Similarity Score": score} for job, company, score in zip(top_n_jobs['jobtitle'], top_n_jobs['company'], top_n_jobs['similarity_score'])]
                    st.table(pd.DataFrame(table_data))

                    #Plot the line chart
                    st.header("Line Chart Showing the Similarity Scores for Recommended Jobs Using Word2Vec")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(top_n_jobs['jobtitle'], top_n_jobs['similarity_score'], marker='o')
                    ax.set_xlabel('Job Title')
                    ax.set_ylabel('Similarity Score')
                    ax.set_title('Similarity Scores for Recommended Jobs')
                    ax.tick_params(axis='x', rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)

if __name__ == "__main__":
    main()
