import streamlit as st
from textblob import TextBlob
import pandas as pd

st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    layout="wide"
)

st.title("AI-Powered Sentiment Analyzer Dashboard")
 
st.markdown("""
### 🧠 About This Project
This dashboard analyzes customer reviews using Natural Language Processing (NLP).

It classifies reviews into:
- 😊 Positive
- ☹️ Negative
- 😐 Neutral

It also shows real-time analytics using interactive charts.
""")

review = st.text_area("Enter a customer review:")

if st.button("Analyze Sentiment"):
    if not review.strip():
        st.warning("Enter a review before analyzing sentiment.")
    else:
        polarity = TextBlob(review).sentiment.polarity

        if polarity > 0:
            st.success("Positive 😊")
        elif polarity < 0:
            st.error("Negative ☹️")
        else:
            st.info("Neutral 😐")

st.header("Dataset Analysis")

data = None

TEXT_COLUMN_CANDIDATES = (
    "review",
    "text",
    "tweet",
    "full_text",
    "content",
    "body",
    "message",
    "caption",
    "comment",
)


def find_text_column(columns):
    normalized_columns = {str(column).strip().lower(): column for column in columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in normalized_columns:
            return normalized_columns[candidate]
    return None

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if st.button("Generate 100 Sample Reviews"):

    reviews = [
        "Great product",
        "Very bad experience",
        "I love it",
        "Not good",
        "Excellent service",
        "Worst product",
        "Amazing quality",
        "Okay product",
        "Highly recommended",
        "Waste of money"
    ] * 10

    data = pd.DataFrame({"review": reviews[:100]})
    st.success("100 sample reviews generated!")

elif uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

if data is not None:

    def get_sentiment(review):
        polarity = TextBlob(review).sentiment.polarity

        if polarity > 0:
            return "Positive"
        elif polarity < 0:
            return "Negative"
        else:
            return "Neutral"

    st.write("Columns in dataset:", data.columns)

    text_column = find_text_column(data.columns)
    if text_column is None:
        st.error(
            "No valid text column found. Use review, text, tweet, full_text, "
            "content, body, message, caption, or comment."
        )
        st.stop()

    text_values = data[text_column].fillna("").astype(str).str.strip()
    if not text_values.astype(bool).any():
        st.error("The selected text column is empty.")
        st.stop()

    data["Sentiment"] = text_values.apply(get_sentiment)

    st.write(data)

    # 📊 STATS
    total = len(data)
    positive = (data["Sentiment"] == "Positive").sum()
    negative = (data["Sentiment"] == "Negative").sum()
    neutral = (data["Sentiment"] == "Neutral").sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total", total)
    col2.metric("Positive 😊", positive)
    col3.metric("Negative ☹️", negative)
    col4.metric("Neutral 😐", neutral)

    # 📊 CHART
    counts = data["Sentiment"].value_counts()
    st.bar_chart(counts)

    # 💾 DOWNLOAD (THIS IS THE CORRECT PART)
    csv_data = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Results CSV",
        data=csv_data,
        file_name="sentiment_results.csv",
        mime="text/csv"
    )
