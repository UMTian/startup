import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

# Download necessary corpora for TextBlob
nltk.download('stopwords')

# Function to calculate sentiment score for a text
def calculate_sentiment_score(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

# Function to generate a word cloud from the text
def generate_word_cloud(text):
    stop_words = set(stopwords.words('english'))
    wordcloud = WordCloud(stopwords=stop_words, background_color='white', width=800, height=400).generate(text)
    return wordcloud

# Function to process the uploaded file and calculate average sentiment score
def process_reviews(file):
    df = pd.read_excel(file)

    if 'Review' not in df.columns:
        st.error("Error: 'Review' column not found in the uploaded file. Please ensure the Excel file contains a 'Review' column.")
        return None, None

    # Combine all reviews into a single text
    all_reviews = ' '.join(df['Review'].astype(str))

    # Calculate sentiment score for each review
    df['Sentiment Score'] = df['Review'].apply(calculate_sentiment_score)

    # Normalize sentiment scores between 0 and 1
    df['Sentiment Score Normalized'] = (df['Sentiment Score'] + 1) / 2

    # Calculate the average sentiment score
    average_sentiment_score = df['Sentiment Score Normalized'].mean()

    # Generate word cloud
    wordcloud = generate_word_cloud(all_reviews)

    return average_sentiment_score, wordcloud

# Load the dataset

df = pd.read_excel('dataset.xlsx')

# Define features and target variable for Success Rate
X = df[['Sentiment Score', 'Valuation', 'Fundings', 'R&D', 'CEO Experience']]
y = df['Success Rate']

# Normalize numerical features
scaler = StandardScaler()
X[['Valuation', 'Fundings', 'R&D', 'CEO Experience']] = scaler.fit_transform(
    X[['Valuation', 'Fundings', 'R&D', 'CEO Experience']])

# Initialize and train Random Forest model
model = RandomForestRegressor()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Sidebar with options
st.sidebar.title('Options')
option = st.sidebar.radio(
    'Select a task:',
    ('Calculate Sentiment Score', 'Calculate Success Rate', 'Calculate Risk', 'Credits')
)

if option == 'Calculate Sentiment Score':
    st.title('Sentiment Analysis for Startups')

    uploaded_file = st.file_uploader("Upload an Excel file with reviews", type=["xlsx"])

    if uploaded_file is not None:
        # Calculate the average sentiment score and generate word cloud
        average_sentiment_score, wordcloud = process_reviews(uploaded_file)

        if average_sentiment_score is not None:
            # Display the average sentiment score as a percentage
            percentage_score = average_sentiment_score
            st.success(f"Sentiment Score: {percentage_score:.2f}")

            # Display the word cloud
            st.image(wordcloud.to_image(), use_column_width=True)

elif option == 'Calculate Success Rate':
    st.title('Predict Success Rate')

    # Dropdown input for industry selection (for visual purposes only)
    industries = ['Technology', 'Healthcare', 'Finance', 'Retail', 'Energy', 'Manufacturing', 'Telecommunications']
    selected_industry = st.selectbox('Select Industry', industries)

    # Input fields
    sentiment_score = st.number_input('Sentiment Score', min_value=0.0, max_value=1.0, value=0.60)
    valuation = st.number_input('Valuation (in billions)', min_value=0.0, value=4.0)
    fundings = st.number_input('Fundings (in billions)', min_value=0.0, value=5.0)
    rd = st.number_input('R&D (in billions)', min_value=0.0, value=0.3)
    ceo_experience = st.number_input('CEO Experience (in years)', min_value=0, value=5)

    # Prepare new data
    new_data = pd.DataFrame({
        'Sentiment Score': [sentiment_score],
        'Valuation': [valuation],
        'Fundings': [fundings],
        'R&D': [rd],
        'CEO Experience': [ceo_experience]
    })

    # Preprocess new data
    new_data[['Valuation', 'Fundings', 'R&D', 'CEO Experience']] = scaler.transform(
        new_data[['Valuation', 'Fundings', 'R&D', 'CEO Experience']])
    X_new = new_data[['Sentiment Score', 'Valuation', 'Fundings', 'R&D', 'CEO Experience']]

    # Make prediction
    if st.button('Predict'):
        y_new_pred = model.predict(X_new)
        prediction = y_new_pred[0]

        # Display the prediction in the form of a pie chart with a baseline of 0%
        labels = ['Success Rate', 'Fail Chance']
        sizes = [prediction, 100 - prediction]  # Adjusting the remaining value for comparison

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#90EE90', '#FF5B61'])
        ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.

        st.pyplot(fig)

elif option == 'Calculate Risk':
    st.title('Calculate Company Risk')

    # Input fields
    loss = st.number_input('Loss (in millions)', min_value=0.0, value=10.0)
    debt = st.number_input('Debt (in millions)', min_value=0.0, value=20.0)
    competitors = st.number_input('Number of Competitors', min_value=0, value=5)
    sentiment_score = st.number_input('Sentiment Score', min_value=0.0, max_value=1.0, value=0.60)

    # Normalization of inputs to 0-1 range
    loss_max = 100.0  # Example maximum value for loss
    debt_max = 200.0  # Example maximum value for debt
    competitors_max = 50.0  # Example maximum value for competitors

    loss_normalized = loss / loss_max
    debt_normalized = debt / debt_max
    competitors_normalized = competitors / competitors_max
    inverse_sentiment = 1 - sentiment_score  # Inverse sentiment score already in 0-1 range

    # Calculate risk using normalized values
    risk = 0.2 * loss_normalized + 0.3 * debt_normalized + 0.1 * competitors_normalized + 0.4 * inverse_sentiment

    # Calculate contributions to risk
    contributions = {
        'Loss': 0.2 * loss_normalized,
        'Debt': 0.3 * debt_normalized,
        'Competitors': 0.1 * competitors_normalized,
        'Inverse Sentiment Score': 0.4 * inverse_sentiment
    }

    if st.button('Calculate Risk'):
        st.write(f"Total Risk: {risk:.2f}")

        # Bar chart to show the contributions to risk
        fig, ax = plt.subplots()
        ax.bar(contributions.keys(), contributions.values(), color=['#FF5B61', '#FFAA61', '#61BFFF', '#90EE90'])
        ax.set_ylabel('Contribution to Risk')
        ax.set_title('Factors Contributing to Risk')

        st.pyplot(fig)

elif option == 'Credits':
    st.title('Credits and Acknowledgments')
    st.write("Thank you for the creation of the dataset. It was nice working with you.")
    st.write("**Credits:**")
    st.write("1. Najeebullah Nawaz")
    st.write("2. Muhammad Naeem")
    st.write("3. Tayyaba Bashir")
