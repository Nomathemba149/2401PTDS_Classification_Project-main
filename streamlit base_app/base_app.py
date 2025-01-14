import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from wordcloud import WordCloud
import string

# Streamlit UI
import streamlit as st

# Corrected title using st.title()
st.title("News Classification with Machine Learning")

# If you want the multi-line description, use st.markdown
st.markdown("""
    - **Objective**: Classify news articles into categories like Business, Technology, Sports, etc.
    - **Data Source**: [Dataset](https://raw.githubusercontent.com/Jana-Liebenberg/2401PTDS_Classification_Project/refs/heads/main/Data/processed/train.csv)
    - **Approach**: Explore EDA, train models (Logistic Regression, Random Forest), and evaluate performance.
""")


# Sidebar for Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select a page", ["Overview", "Data Preprocessing", "Model Training & Evaluation", "Word Cloud", "Top Frequent Words", "Predict Category"])

# Page: Data Preprocessing
if page == "Data Preprocessing":
    uploaded_file = st.file_uploader("Upload Your File", type=["csv", "xlsx", "json"])
    
    if uploaded_file is not None:
        # Try to read the file based on its extension
        file_extension = uploaded_file.name.split('.')[-1]

        if file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension == "xlsx":
            df = pd.read_excel(uploaded_file)
        elif file_extension == "json":
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file format!")

        # Data Pre-processing
        st.write("### Data Cleaning & Preprocessing")

        # Drop 'url' column if exists
        if 'url' in df.columns:
            df.drop(columns=['url'], inplace=True)

        # Combine text columns (if available) and clean them
        if 'headlines' in df.columns and 'description' in df.columns and 'content' in df.columns:
            df['combined_text'] = df['headlines'] + ' ' + df['description'] + ' ' + df['content']
            df['combined_text'] = df['combined_text'].str.lower()
            df['combined_text'] = df['combined_text'].apply(lambda x: ''.join([l for l in x if l not in string.punctuation + '0123456789']))
        else:
            st.error("Required text columns ('headlines', 'description', 'content') not found in the dataset.")
        
        # Show a preview of the cleaned data
        st.write(df[['combined_text']].head())

        # Store the cleaned data in session state
        st.session_state.df = df

# Page: Model Training & Evaluation
if page == "Model Training & Evaluation":
    # Ensure data has been uploaded and preprocessed before proceeding
    if 'df' in st.session_state:
        df = st.session_state.df  # Retrieve the data from session state

        # Label encoding for target variable
        if 'category' in df.columns:
            label_encoder = LabelEncoder()
            df['category_encoded'] = label_encoder.fit_transform(df['category'])
            st.session_state.label_encoder = label_encoder
        else:
            st.error("Category column not found in the dataset.")
        
        # Vectorization
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        X_tfidf = vectorizer.fit_transform(df['combined_text'])
        y = df['category_encoded']
        
        # Store the vectorizer in session state
        st.session_state.vectorizer = vectorizer

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

        # Train Logistic Regression model
        log_reg = LogisticRegression(max_iter=1000, random_state=42)
        log_reg.fit(X_train, y_train)

        # Store the model in session state
        st.session_state.log_reg = log_reg

        # Predictions and Evaluation
        y_pred_log_reg = log_reg.predict(X_test)
        log_reg_accuracy = accuracy_score(y_test, y_pred_log_reg)
        log_reg_report = classification_report(y_test, y_pred_log_reg)

        # Train Random Forest Classifier
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_train)

        # Predictions and Evaluation
        y_pred_rf = rf_classifier.predict(X_test)
        rf_accuracy = accuracy_score(y_test, y_pred_rf)
        rf_report = classification_report(y_test, y_pred_rf)

        # Display model evaluation
        st.write("### Model Evaluation")
        st.write(f"Logistic Regression Accuracy: {log_reg_accuracy:.2f}")
        st.text(log_reg_report)
        st.write(f"Random Forest Accuracy: {rf_accuracy:.2f}")
        st.text(rf_report)
    else:
        st.error("Please upload and preprocess the data first!")

# Page: Word Cloud
if page == "Word Cloud":
    # Ensure that data has been uploaded and preprocessed before proceeding
    if 'df' in st.session_state and 'vectorizer' in st.session_state:
        df = st.session_state.df  # Retrieve the data from session state
        st.write("### Word Cloud for Frequent Terms")
        text_data = ' '.join(df['combined_text'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    else:
        st.error("Please upload and preprocess the data first!")

# Page: Top Frequent Words
if page == "Top Frequent Words":
    # Ensure that data has been uploaded and preprocessed before proceeding
    if 'df' in st.session_state and 'vectorizer' in st.session_state:
        df = st.session_state.df  # Retrieve the data from session state
        st.write("### Top 10 Most Frequent Words")
        vectorizer = st.session_state.vectorizer  # Retrieve the vectorizer from session state
        word_counts = vectorizer.transform(df['combined_text']).toarray()
        word_freq = dict(zip(vectorizer.get_feature_names_out(), word_counts.sum(axis=0)))
        sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        words, counts = zip(*sorted_word_freq)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(words), y=list(counts))
        plt.title('Top 10 Most Frequent Words')
        plt.xticks(rotation=45)
        st.pyplot(plt)
    else:
        st.error("Please upload and preprocess the data first!")

# Page: Predict Category
if page == "Predict Category":
    st.write("### Predict the Category of a News Article")
    user_input = st.text_area("Enter a news article content:")
    if user_input:
        # Ensure that data and models are ready for prediction
        if 'df' in st.session_state and 'vectorizer' in st.session_state and 'log_reg' in st.session_state and 'label_encoder' in st.session_state:
            input_tfidf = st.session_state.vectorizer.transform([user_input])
            prediction = st.session_state.log_reg.predict(input_tfidf)
            predicted_category = st.session_state.label_encoder.inverse_transform(prediction)
            st.write(f"Predicted Category: {predicted_category[0]}")
        else:
            st.error("Please upload and preprocess the data, and train the model first!")

