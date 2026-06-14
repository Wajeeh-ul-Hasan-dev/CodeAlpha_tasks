# ============================================================
# Twitter / Social Media — Sentiment Analysis
# CodeAlpha Data Analytics Internship — Task 4
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
import re
import warnings

warnings.filterwarnings('ignore')

# NLP Libraries
try:
    from textblob import TextBlob
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'textblob', '-q'])
    from textblob import TextBlob

try:
    from wordcloud import WordCloud
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'wordcloud', '-q'])
    from wordcloud import WordCloud

# ─────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────
# Using Sentiment140 dataset (1.6M tweets) — public domain
# Source: https://www.kaggle.com/datasets/kazanova/sentiment140
# If you have the CSV: df = pd.read_csv('training.1600000.processed.noemoticon.csv', ...)

url = "https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment/master/Tweets.csv"

try:
    df = pd.read_csv(url)
    print("✅ Dataset loaded: Twitter US Airline Sentiment")
    text_col = 'text'
    if 'airline_sentiment' in df.columns:
        df['sentiment_label'] = df['airline_sentiment']
    else:
        df['sentiment_label'] = None
except:
    print("⚠️ Could not load from URL. Generating sample tweet dataset...")
    import random
    random.seed(42)

    sample_tweets = [
        # Positive
        "Just had an amazing flight! Great service and on time arrival 🎉",
        "Loving the new update! This app is fantastic and super easy to use.",
        "Best customer service ever. They really went above and beyond!",
        "Beautiful day today, feeling grateful and happy 😊",
        "Great experience overall. Will definitely recommend to friends!",
        "The food was absolutely delicious. 10/10 would visit again.",
        "So proud of my team today. We crushed it! 🔥",
        "Finally got my package. Fast delivery, great quality product!",
        "This movie was incredible. One of the best I've seen this year.",
        "Shoutout to the support team — they fixed my issue in minutes!",
        # Negative
        "Terrible flight experience. Delayed for 3 hours with no explanation.",
        "Worst customer service I've ever dealt with. Completely useless.",
        "The app keeps crashing. This is so frustrating and annoying.",
        "Never buying from this brand again. Total waste of money.",
        "Flight cancelled last minute. No compensation, no apology. Disgusting.",
        "Horrible experience. Staff was rude and unhelpful throughout.",
        "My order arrived broken. Support ignored my complaint for days.",
        "This product is garbage. Stopped working after two days.",
        "So disappointed. Expected so much better from this company.",
        "Lost my luggage and nobody seems to care. Absolutely furious.",
        # Neutral
        "Just landed at JFK airport. Heading to the hotel.",
        "The flight was okay. Nothing special but got there on time.",
        "Using the app for the first time. Let's see how it goes.",
        "Checked in to the hotel. Room is standard.",
        "Watching the news right now. Lots happening today.",
        "The meeting was average. Some good points, some not so much.",
        "Got my order today. It's what I expected.",
        "Weather is cloudy. Might rain later.",
        "Tried the new restaurant. It was decent, nothing extraordinary.",
        "Flight boarded on time. Currently in the air.",
    ]

    n = 1000
    tweets = random.choices(sample_tweets, k=n)
    df = pd.DataFrame({'text': tweets, 'sentiment_label': None})
    text_col = 'text'

print(f"📊 Total Tweets Loaded: {len(df)}")

# ─────────────────────────────────────────
# 2. TEXT CLEANING
# ─────────────────────────────────────────
def clean_tweet(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)       # Remove URLs
    text = re.sub(r'@\w+', '', text)                  # Remove mentions
    text = re.sub(r'#(\w+)', r'\1', text)             # Remove # but keep word
    text = re.sub(r'[^a-z\s]', '', text)              # Remove special chars
    text = re.sub(r'\s+', ' ', text).strip()          # Remove extra spaces
    return text

df['clean_text'] = df[text_col].apply(clean_tweet)
print("✅ Text cleaning done.")

# ─────────────────────────────────────────
# 3. SENTIMENT ANALYSIS (TextBlob)
# ─────────────────────────────────────────
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0.05:
        return 'Positive', polarity
    elif polarity < -0.05:
        return 'Negative', polarity
    else:
        return 'Neutral', polarity

df[['sentiment', 'polarity']] = df['clean_text'].apply(
    lambda x: pd.Series(get_sentiment(x))
)

# Subjectivity
df['subjectivity'] = df['clean_text'].apply(
    lambda x: TextBlob(x).sentiment.subjectivity
)

print("✅ Sentiment analysis complete.")
print("\n📌 Sentiment Distribution:")
print(df['sentiment'].value_counts())

# ─────────────────────────────────────────
# 4. VISUALIZATIONS
# ─────────────────────────────────────────
DARK_BG = '#0d1117'
POSITIVE = '#2ecc71'
NEGATIVE = '#e74c3c'
NEUTRAL = '#3498db'
TEXT_COLOR = 'white'

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor': '#161b22',
    'axes.labelcolor': TEXT_COLOR,
    'xtick.color': TEXT_COLOR,
    'ytick.color': TEXT_COLOR,
    'text.color': TEXT_COLOR,
    'grid.color': '#21262d',
    'grid.alpha': 0.6
})

sentiment_colors = {'Positive': POSITIVE, 'Negative': NEGATIVE, 'Neutral': NEUTRAL}
color_list = [sentiment_colors[s] for s in df['sentiment'].value_counts().index]

fig = plt.figure(figsize=(18, 18))
fig.suptitle('Twitter Sentiment Analysis', fontsize=24,
             fontweight='bold', color='white', y=0.98)

# ── Plot 1: Sentiment Distribution (Bar) ──
ax1 = fig.add_subplot(3, 3, 1)
sentiment_counts = df['sentiment'].value_counts()
bars = ax1.bar(sentiment_counts.index, sentiment_counts.values,
               color=color_list, edgecolor='none', width=0.5)
for bar, val in zip(bars, sentiment_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(val), ha='center', color='white', fontsize=11, fontweight='bold')
ax1.set_title('Sentiment Distribution', color='white', fontsize=13)
ax1.set_ylabel('Tweet Count')

# ── Plot 2: Sentiment Pie Chart ──
ax2 = fig.add_subplot(3, 3, 2)
wedges, texts, autotexts = ax2.pie(
    sentiment_counts.values, labels=sentiment_counts.index,
    autopct='%1.1f%%', colors=color_list, startangle=140,
    textprops={'color': 'white', 'fontsize': 11}
)
for at in autotexts:
    at.set_fontweight('bold')
    at.set_fontsize(12)
ax2.set_title('Sentiment Share', color='white', fontsize=13)

# ── Plot 3: Polarity Distribution ──
ax3 = fig.add_subplot(3, 3, 3)
ax3.hist(df[df['sentiment'] == 'Positive']['polarity'], bins=30,
         color=POSITIVE, alpha=0.7, label='Positive')
ax3.hist(df[df['sentiment'] == 'Negative']['polarity'], bins=30,
         color=NEGATIVE, alpha=0.7, label='Negative')
ax3.hist(df[df['sentiment'] == 'Neutral']['polarity'], bins=30,
         color=NEUTRAL, alpha=0.7, label='Neutral')
ax3.axvline(0, color='white', linestyle='--', linewidth=1)
ax3.set_title('Polarity Score Distribution', color='white', fontsize=13)
ax3.set_xlabel('Polarity (-1 to +1)')
ax3.set_ylabel('Frequency')
ax3.legend(facecolor='#161b22', edgecolor='gray', labelcolor='white')

# ── Plot 4: Subjectivity vs Polarity (Scatter) ──
ax4 = fig.add_subplot(3, 3, 4)
sample = df.sample(min(500, len(df)), random_state=42)
for sentiment, color in sentiment_colors.items():
    mask = sample['sentiment'] == sentiment
    ax4.scatter(sample[mask]['polarity'], sample[mask]['subjectivity'],
                c=color, alpha=0.5, s=20, label=sentiment)
ax4.set_title('Polarity vs Subjectivity', color='white', fontsize=13)
ax4.set_xlabel('Polarity')
ax4.set_ylabel('Subjectivity')
ax4.legend(facecolor='#161b22', edgecolor='gray', labelcolor='white')
ax4.axvline(0, color='white', linestyle='--', linewidth=0.8, alpha=0.5)
ax4.axhline(0.5, color='white', linestyle='--', linewidth=0.8, alpha=0.5)

# ── Plot 5: Average Polarity by Sentiment ──
ax5 = fig.add_subplot(3, 3, 5)
avg_polarity = df.groupby('sentiment')['polarity'].mean().reindex(['Positive', 'Neutral', 'Negative'])
bars = ax5.bar(avg_polarity.index, avg_polarity.values,
               color=[POSITIVE, NEUTRAL, NEGATIVE], edgecolor='none', width=0.5)
for bar, val in zip(bars, avg_polarity.values):
    ax5.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.005 if val >= 0 else bar.get_height() - 0.02,
             f'{val:.3f}', ha='center', color='white', fontsize=11, fontweight='bold')
ax5.set_title('Avg Polarity by Sentiment', color='white', fontsize=13)
ax5.set_ylabel('Average Polarity Score')
ax5.axhline(0, color='white', linestyle='--', linewidth=0.8)

# ── Plot 6: Subjectivity Distribution ──
ax6 = fig.add_subplot(3, 3, 6)
for sentiment, color in sentiment_colors.items():
    subset = df[df['sentiment'] == sentiment]['subjectivity']
    ax6.hist(subset, bins=25, color=color, alpha=0.6, label=sentiment)
ax6.set_title('Subjectivity Distribution', color='white', fontsize=13)
ax6.set_xlabel('Subjectivity (0=Objective, 1=Subjective)')
ax6.set_ylabel('Frequency')
ax6.legend(facecolor='#161b22', edgecolor='gray', labelcolor='white')

# ── Plot 7: Word Cloud (Positive Tweets) ──
ax7 = fig.add_subplot(3, 3, 7)
positive_text = ' '.join(df[df['sentiment'] == 'Positive']['clean_text'])
if positive_text.strip():
    wc = WordCloud(width=400, height=300, background_color='#161b22',
                   colormap='Greens', max_words=80,
                   stopwords={'the', 'a', 'an', 'is', 'it', 'to', 'and', 'of',
                               'for', 'in', 'on', 'with', 'was', 'this', 'that'}).generate(positive_text)
    ax7.imshow(wc, interpolation='bilinear')
ax7.axis('off')
ax7.set_title('Positive Tweets — Word Cloud', color=POSITIVE, fontsize=13)

# ── Plot 8: Word Cloud (Negative Tweets) ──
ax8 = fig.add_subplot(3, 3, 8)
negative_text = ' '.join(df[df['sentiment'] == 'Negative']['clean_text'])
if negative_text.strip():
    wc = WordCloud(width=400, height=300, background_color='#161b22',
                   colormap='Reds', max_words=80,
                   stopwords={'the', 'a', 'an', 'is', 'it', 'to', 'and', 'of',
                               'for', 'in', 'on', 'with', 'was', 'this', 'that'}).generate(negative_text)
    ax8.imshow(wc, interpolation='bilinear')
ax8.axis('off')
ax8.set_title('Negative Tweets — Word Cloud', color=NEGATIVE, fontsize=13)

# ── Plot 9: Top 10 Most Common Words ──
ax9 = fig.add_subplot(3, 3, 9)
stopwords = {'the', 'a', 'an', 'is', 'it', 'to', 'and', 'of', 'for',
             'in', 'on', 'with', 'was', 'this', 'that', 'i', 'my', 'we',
             'are', 'be', 'at', 'so', 'im', 'just', 'have', 'me', 'get'}
all_words = []
for text in df['clean_text']:
    words = [w for w in text.split() if w not in stopwords and len(w) > 2]
    all_words.extend(words)
top_words = pd.Series(Counter(all_words)).sort_values(ascending=False).head(10)
ax9.barh(top_words.index[::-1], top_words.values[::-1],
         color='#3498db', edgecolor='none', alpha=0.85)
ax9.set_title('Top 10 Most Common Words', color='white', fontsize=13)
ax9.set_xlabel('Frequency')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('twitter_sentiment.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.show()
print("\n✅ Visualization saved as 'twitter_sentiment.png'")

# ─────────────────────────────────────────
# 5. SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("        TWITTER SENTIMENT — KEY INSIGHTS")
print("=" * 55)
print(f"  ✔ Total tweets analyzed       : {len(df):,}")
for s in ['Positive', 'Negative', 'Neutral']:
    count = (df['sentiment'] == s).sum()
    pct = count / len(df) * 100
    print(f"  ✔ {s:<10} tweets         : {count:,} ({pct:.1f}%)")
print(f"  ✔ Avg polarity score          : {df['polarity'].mean():.3f}")
print(f"  ✔ Avg subjectivity score      : {df['subjectivity'].mean():.3f}")
print(f"  ✔ Most common word            : {top_words.index[0]}")
print("=" * 55)
print("  Task 4 Complete. Ready for GitHub upload!")
print("=" * 55)
