# import os
# import fitz  # PyMuPDF
# import pandas as pd
# import numpy as np
# import nltk
# from flask import Flask, request, render_template, redirect, url_for
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# from imblearn.under_sampling import RandomUnderSampler
# from nltk.corpus import stopwords

# # Download stopwords
# nltk.download("stopwords")

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# # Load and preprocess the dataset
# def load_data():
#     data = pd.read_csv("fake_job_posting.csv")
#     data.fillna(' ', inplace=True)
#     stop_words = set(stopwords.words("english"))
#     data['text'] = data['title'] + ' ' + data['location'] + ' ' + data['company_profile'] + ' ' + data['description'] + ' ' + data['requirements'] + ' ' + data['benefits'] + ' ' + data['industry']
#     data.drop(['job_id', 'title', 'location', 'department', 'company_profile', 'description', 'requirements', 'benefits', 'required_experience', 'required_education', 'industry', 'function', 'employment_type'], axis=1, inplace=True)
#     data['text'] = data['text'].apply(lambda x: x.lower())
#     data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

#     # Undersample the data
#     X = data['text']
#     y = data['fraudulent']
#     under_sampler = RandomUnderSampler()
#     X_res, y_res = under_sampler.fit_resample(X.values.reshape(-1, 1), y)
#     data_resampled = pd.DataFrame({'text': X_res.flatten(), 'fraudulent': y_res})

#     return data_resampled

# # Train the model
# def train_model():
#     data = load_data()
#     X_train, X_test, y_train, y_test = train_test_split(data['text'], data['fraudulent'], test_size=0.3, random_state=42)

#     pipeline = Pipeline([
#         ('tfidf', TfidfVectorizer(stop_words='english')),
#         ('clf', LogisticRegression(C=10))
#     ])

#     pipeline.fit(X_train, y_train)
#     return pipeline

# # Load the trained model
# model = train_model()

# # Function to check if uploaded file is allowed
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# # Extract text from PDF
# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with fitz.open(pdf_path) as pdf:
#         for page_num in range(pdf.page_count):
#             page = pdf[page_num]
#             text += page.get_text()
#     return text

# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files['file']
#         if file.filename == '':
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#             file.save(filepath)
#             text = extract_text_from_pdf(filepath)
#             prediction = model.predict([text])[0]
#             result = "FAKE" if prediction == 1 else "REAL"
#             return render_template('result.html', result=result)
#     return render_template('index.html')

# if __name__ == "__main__":
#     app.run(debug=True)


import os
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import nltk
from flask import Flask, request, render_template, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from nltk.corpus import stopwords

# Download stopwords
nltk.download("stopwords")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Load and preprocess the dataset
def load_data():
    data = pd.read_csv("fake_job_posting.csv")
    data.fillna(' ', inplace=True)
    stop_words = set(stopwords.words("english"))
    data['text'] = (data['title'] + ' ' + data['location'] + ' ' + data['company_profile'] + ' ' + 
                    data['description'] + ' ' + data['requirements'] + ' ' + data['benefits'] + ' ' + data['industry'])
    data.drop(['job_id', 'title', 'location', 'department', 'company_profile', 'description', 
               'requirements', 'benefits', 'required_experience', 'required_education', 'industry', 
               'function', 'employment_type'], axis=1, inplace=True)
    data['text'] = data['text'].apply(lambda x: x.lower())
    data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    # Undersample the data
    X = data['text']
    y = data['fraudulent']
    under_sampler = RandomUnderSampler()
    X_res, y_res = under_sampler.fit_resample(X.values.reshape(-1, 1), y)
    data_resampled = pd.DataFrame({'text': X_res.flatten(), 'fraudulent': y_res})

    return data_resampled

# Train the model
def train_model():
    data = load_data()
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['fraudulent'], test_size=0.3, random_state=42)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(C=10))
    ])

    pipeline.fit(X_train, y_train)
    return pipeline

# Load the trained model
model = train_model()

# Function to check if uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Extract text from PDF and check for job-related keywords
def extract_text_from_pdf(pdf_path, keywords):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text()
    text = text.lower()
    keyword_count = sum(text.count(keyword) for keyword in keywords)
    return text if keyword_count >= 3 else None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Define job-related keywords
            job_keywords = ['job', 'position', 'hiring', 'vacancy', 'employment','salary','experience', 'recruitment','recruiting', 'career', 'opportunity']

            text = extract_text_from_pdf(filepath, job_keywords)
            if text:
                prediction = model.predict([text])[0]
                result = "FAKE" if prediction == 1 else "REAL"
            else:
                result = "UNRELATED CONTENT"
            return render_template('result.html', result=result)

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
