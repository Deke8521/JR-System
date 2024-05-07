import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import matplotlib.pyplot as plt
import seaborn as sns
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
        
        job_dataset = job_data[['company', 'jobid', 'jobtitle','jobdescription', 'skills', 'advertiserurl']].copy()
        # Check for null values in specific columns
        null_values = job_dataset[['company', 'jobid', 'jobtitle', 'jobdescription', 'skills', 'advertiserurl']].isnull().sum()
        print("Null values in each column:")
        print(null_values)
        # Remove NA values from job profile
        job_dataset.dropna(inplace=True)
        # Combine job skills
        job_dataset['Skills'] = job_dataset['skills'] + ';' + job_dataset['jobdescription']  # Combine 'skills' and 'jobdescription' columns
        # Drop redundant columns
        job_dataset.drop(columns=['skills', 'jobdescription'], inplace=True)
        # Apply clean_text function to the entire job_dataset excluding 'advertiserurl'
        columns_to_clean = ['company', 'jobid', 'jobtitle', 'Skills']
        for column in columns_to_clean:
            if column != 'advertiserurl':
                job_dataset[column] = job_dataset[column].map(clean_text)
        return job_dataset 
    except Exception as e:
        st.error("Error loading job dataset: " + str(e))
        return None

# Main function
def main():
    st.image("https://raw.githubusercontent.com/Deke8521/JR-System/main/Screenshot%202024-04-28%20205652.png")
   
    st.title("Job Recommendation App")
    
    
    job_dataset = load_job_dataset()


    # Initialize resume_text variable
    resume_text = ""

    # Get input method for resume
    resume_input_method = st.selectbox("Select method to input resume (PDF or Word doc.):", ("Upload PDF", "Upload Word Document"))
    
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
            # Vectorize resume and job descriptions
            
            
                vectorizer = CountVectorizer()
                job_descriptions = job_dataset['Skills'].tolist()
                job_descriptions_count = vectorizer.fit_transform(job_descriptions)
                
                user_resume_vector = vectorizer.transform([resume_text])
              

                
                

                # Calculate cosine similarity between resume and job descriptions
                similarities = cosine_similarity(user_resume_vector, job_descriptions_count)
                print("Distances between user's resume and the first 5 job descriptions:")
                for i in range(5):
                    print(f"Job {i+1}: {similarities[0, i]}")

            
                # Rank job titles based on similarity scores
                
                # Rank job titles based on similarity scores
                recommended_jobs = [(job_dataset.iloc[i]['jobtitle'], job_dataset.iloc[i]['company'], job_dataset.iloc[i]['advertiserurl'], similarities[0][i]) for i in range(len(job_descriptions))]

                recommended_jobs.sort(key=lambda x: x[3], reverse=True)

                sorted_jobs = sorted(recommended_jobs, key=lambda x: x[3], reverse=True)[:10]
                # Calculate average similarity score
                total_similarity_score = sum(score for _, _, _, score in sorted_jobs)
                average_similarity_score = total_similarity_score / len(sorted_jobs)
                st.write(f"Average Similarity Score: {average_similarity_score:.4f}")
               
                # Display recommended jobs in a table
                st.subheader("Top 10 Recommended Jobs using Count Vectorizer:")
                table_data = [{"Job Title": job, "Company": company, "Advertiser URL": url} for job, company, url, _ in sorted_jobs]
                st.table(pd.DataFrame(table_data, index=range(1, 11)))


                # Plot the line plot
                st.header("Line Plot Showing the Similarity Scores for Recommended Jobs using Count Vectorizer")
                job_titles = [job for job, _, _, _ in sorted_jobs]
                similarity_scores = [score for _, _, _,  score in sorted_jobs]
                recommended_jobs_df = pd.DataFrame({'Job Title': job_titles, 'Similarity Score': similarity_scores})

    # Create a line plot
                # Create a line plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(recommended_jobs_df['Job Title'], recommended_jobs_df['Similarity Score'], marker='o')

                # Set labels and title
                ax.set_xlabel('Job Title')
                ax.set_ylabel('Similarity Score')
                ax.set_title('Similarity Scores for Recommended Jobs using Count Vectorizer')

                # Rotate x-axis labels for better readability
                ax.tick_params(axis='x', rotation=45)

                # Ensure tight layout
                plt.tight_layout()

                # Display the plot
                st.pyplot(fig)


if __name__ == "__main__":
    main()


