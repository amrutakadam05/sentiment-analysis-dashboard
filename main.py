import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

data = pd.read_csv("reviews.csv")

def get_sentiment(review):
    polarity = TextBlob(review).sentiment.polarity

    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

data["Sentiment"] = data["review"].apply(get_sentiment)

print(data)

print("\nSummary:")
print(data["Sentiment"].value_counts())

data.to_csv("results.csv", index=False)
print("Results saved to results.csv")

counts = data["Sentiment"].value_counts()

counts.plot(kind="bar")

plt.title("Sentiment Analysis Results")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.show()