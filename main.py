import requests
from bs4 import BeautifulSoup
from pathlib import Path
import pandas as pd
from textblob import TextBlob
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Get the ECB page
url = "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/html/ecb.is251030~4f74dde15e.en.html"

response = requests.get(url, timeout=15)
soup = BeautifulSoup(response.text, "html.parser")

# Extract the main text
container = soup.select_one("div#main-wrapper main")
text = container.get_text(separator="\n\n", strip=True)

# Save the text file
Path("output").mkdir(exist_ok=True)

text_path = Path("output/ecb_pressconf_2025_10_30.txt")
text_path.write_text(text, encoding="utf-8")

# Paragraph-level sentiment analysis
paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]

sentiment_results = []

for i, paragraph in enumerate(paragraphs, 1):
    polarity = TextBlob(paragraph).sentiment.polarity

    if polarity > 0.1:
        label = "positive"
    elif polarity < -0.1:
        label = "negative"
    else:
        label = "neutral"

    sentiment_results.append({
        "paragraph_number": i,
        "paragraph_text": paragraph,
        "sentiment_score": polarity,
        "sentiment_label": label
    })

sentiment_df = pd.DataFrame(sentiment_results)
sentiment_df.to_csv("output/paragraph_sentiment.csv", index=False)

# Word frequency analysis
stop_words = set([
    "the", "and", "to", "of", "in", "a", "is", "that", "we", "our",
    "for", "by", "on", "with", "as", "at", "from", "this", "are",
    "be", "has", "have", "it", "will", "an", "or", "was", "were",
    "but", "not", "can", "you", "i", "he", "she", "they", "them",
    "their", "what", "which", "who", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "too", "very", "s", "t", "just",
    "don", "should", "now"
])

words = re.findall(r"\b\w+\b", text.lower())
filtered_words = [word for word in words if word not in stop_words and len(word) > 2]

word_counts = Counter(filtered_words).most_common(50)

word_df = pd.DataFrame(word_counts, columns=["word", "frequency"])
word_df.to_csv("output/word_frequencies.csv", index=False)

# Word cloud
wordcloud = WordCloud(width=800, height=400, background_color="white")
wordcloud = wordcloud.generate_from_frequencies(dict(word_counts))

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()
plt.savefig("output/wordcloud.png")
plt.close()

print("Saved text to", text_path)
print("Saved sentiment results to output/paragraph_sentiment.csv")
print("Saved word frequencies to output/word_frequencies.csv")
print("Saved word cloud to output/wordcloud.png")