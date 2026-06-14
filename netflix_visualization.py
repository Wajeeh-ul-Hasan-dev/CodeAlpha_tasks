# ============================================================
# Netflix Movies & TV Shows — Data Visualization
# CodeAlpha Data Analytics Internship — Task 3
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────
# Dataset: Netflix Movies and TV Shows (built from public data)
# Source: https://www.kaggle.com/datasets/shivamb/netflix-shows
# If you have the CSV, replace the URL below with your file path:
# df = pd.read_csv('netflix_titles.csv')

url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/netflix_titles.csv"

try:
    df = pd.read_csv(url)
    print("✅ Dataset loaded from URL.")
except:
    # Fallback: generate a representative sample dataset
    print("⚠️ Could not load from URL. Generating sample dataset...")
    import random
    random.seed(42)
    np.random.seed(42)

    types = ['Movie'] * 6131 + ['TV Show'] * 2676
    ratings = ['TV-MA'] * 3207 + ['TV-14'] * 2160 + ['TV-PG'] * 863 + \
              ['R'] * 799 + ['PG-13'] * 490 + ['NR'] * 550 + ['PG'] * 369 + ['G'] * 135
    countries = ['United States'] * 3690 + ['India'] * 972 + ['United Kingdom'] * 419 + \
                ['Japan'] * 245 + ['South Korea'] * 199 + ['Canada'] * 181 + \
                ['France'] * 123 + ['Spain'] * 145 + ['Mexico'] * 110 + ['Other'] * 2723
    years = list(range(2008, 2022))
    genres = ['Dramas', 'Comedies', 'Documentaries', 'Action & Adventure',
              'Thrillers', 'Horror Movies', 'Romantic Movies', 'Children & Family',
              'International Movies', 'Stand-Up Comedy']

    n = 8807
    df = pd.DataFrame({
        'type': random.choices(types, k=n),
        'rating': random.choices(ratings, k=n),
        'country': random.choices(countries, k=n),
        'release_year': np.random.choice(years, n),
        'listed_in': random.choices(genres, k=n),
        'date_added': pd.date_range('2008-01-01', periods=n, freq='8H').strftime('%B %d, %Y'),
        'duration': [f"{random.randint(70, 160)} min" if t == 'Movie'
                     else f"{random.randint(1, 10)} Season{'s' if random.random() > 0.4 else ''}"
                     for t in random.choices(types, k=n)]
    })

# ─────────────────────────────────────────
# 2. BASIC CLEANING
# ─────────────────────────────────────────
df.dropna(subset=['type', 'release_year'], inplace=True)
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['year_added'] = df['date_added'].dt.year
df['release_year'] = df['release_year'].astype(int)

print(f"📊 Total Titles: {len(df)}")
print(f"📽️  Movies: {(df['type'] == 'Movie').sum()}")
print(f"📺  TV Shows: {(df['type'] == 'TV Show').sum()}")

# ─────────────────────────────────────────
# 3. VISUALIZATIONS
# ─────────────────────────────────────────
# Netflix color palette
NETFLIX_RED = '#E50914'
NETFLIX_DARK = '#141414'
NETFLIX_GRAY = '#564d4d'
LIGHT_RED = '#ff6b6b'
LIGHT_GRAY = '#b3b3b3'

plt.rcParams.update({
    'figure.facecolor': NETFLIX_DARK,
    'axes.facecolor': '#1a1a1a',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'text.color': 'white',
    'grid.color': '#333333',
    'grid.alpha': 0.5
})

fig = plt.figure(figsize=(18, 20))
fig.suptitle('Netflix Content Analysis', fontsize=24, fontweight='bold',
             color=NETFLIX_RED, y=0.98)

# ── Plot 1: Movies vs TV Shows (Pie) ──
ax1 = fig.add_subplot(3, 3, 1)
type_counts = df['type'].value_counts()
colors = [NETFLIX_RED, LIGHT_GRAY]
wedges, texts, autotexts = ax1.pie(
    type_counts, labels=type_counts.index, autopct='%1.1f%%',
    colors=colors, startangle=90,
    textprops={'color': 'white', 'fontsize': 11}
)
for at in autotexts:
    at.set_fontsize(12)
    at.set_fontweight('bold')
ax1.set_title('Movies vs TV Shows', color='white', fontsize=13, pad=10)

# ── Plot 2: Content Added Per Year ──
ax2 = fig.add_subplot(3, 3, 2)
if 'year_added' in df.columns:
    yearly = df['year_added'].dropna().astype(int).value_counts().sort_index()
    yearly = yearly[yearly.index >= 2010]
    bars = ax2.bar(yearly.index, yearly.values, color=NETFLIX_RED, edgecolor='none', alpha=0.9)
    ax2.bar(yearly.index[yearly.values == yearly.values.max()],
            yearly.values.max(), color='white', alpha=0.9)
    ax2.set_title('Content Added Per Year', color='white', fontsize=13)
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Titles Added')
    ax2.tick_params(axis='x', rotation=45)

# ── Plot 3: Top 10 Countries ──
ax3 = fig.add_subplot(3, 3, 3)
country_counts = df['country'].dropna().value_counts().head(10)
bars = ax3.barh(country_counts.index[::-1], country_counts.values[::-1],
                color=[NETFLIX_RED if i == 9 else LIGHT_GRAY for i in range(10)],
                edgecolor='none')
ax3.set_title('Top 10 Countries by Content', color='white', fontsize=13)
ax3.set_xlabel('Number of Titles')

# ── Plot 4: Rating Distribution ──
ax4 = fig.add_subplot(3, 3, 4)
rating_order = df['rating'].dropna().value_counts().head(8)
colors_rating = [NETFLIX_RED if i == 0 else '#ff4444' if i == 1 else LIGHT_GRAY
                 for i in range(len(rating_order))]
ax4.bar(rating_order.index, rating_order.values, color=colors_rating, edgecolor='none')
ax4.set_title('Content Rating Distribution', color='white', fontsize=13)
ax4.set_xlabel('Rating')
ax4.set_ylabel('Count')
ax4.tick_params(axis='x', rotation=30)

# ── Plot 5: Release Year Distribution ──
ax5 = fig.add_subplot(3, 3, 5)
release = df['release_year'].value_counts().sort_index()
release = release[release.index >= 1990]
ax5.plot(release.index, release.values, color=NETFLIX_RED, linewidth=2.5)
ax5.fill_between(release.index, release.values, alpha=0.2, color=NETFLIX_RED)
ax5.set_title('Titles by Release Year', color='white', fontsize=13)
ax5.set_xlabel('Year')
ax5.set_ylabel('Number of Titles')

# ── Plot 6: Top Genres ──
ax6 = fig.add_subplot(3, 3, 6)
all_genres = []
for g in df['listed_in'].dropna():
    all_genres.extend([x.strip() for x in g.split(',')])
genre_counts = pd.Series(Counter(all_genres)).sort_values(ascending=False).head(10)
ax6.barh(genre_counts.index[::-1], genre_counts.values[::-1],
         color=NETFLIX_RED, edgecolor='none', alpha=0.85)
ax6.set_title('Top 10 Genres', color='white', fontsize=13)
ax6.set_xlabel('Count')

# ── Plot 7: Movies vs TV Shows Over Years ──
ax7 = fig.add_subplot(3, 1, 3)
if 'year_added' in df.columns:
    pivot = df.groupby(['year_added', 'type']).size().unstack(fill_value=0)
    pivot = pivot[pivot.index >= 2010]
    if 'Movie' in pivot.columns:
        ax7.plot(pivot.index, pivot['Movie'], color=NETFLIX_RED,
                 linewidth=2.5, label='Movies', marker='o', markersize=5)
    if 'TV Show' in pivot.columns:
        ax7.plot(pivot.index, pivot['TV Show'], color=LIGHT_GRAY,
                 linewidth=2.5, label='TV Shows', marker='s', markersize=5)
    ax7.set_title('Movies vs TV Shows Added Over Time', color='white', fontsize=13)
    ax7.set_xlabel('Year')
    ax7.set_ylabel('Titles Added')
    ax7.legend(facecolor='#1a1a1a', edgecolor='gray', labelcolor='white')
    ax7.tick_params(axis='x', rotation=45)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('netflix_visualization.png', dpi=150, bbox_inches='tight',
            facecolor=NETFLIX_DARK)
plt.show()
print("\n✅ Visualization saved as 'netflix_visualization.png'")

# ─────────────────────────────────────────
# 4. SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("           NETFLIX DATA — KEY INSIGHTS")
print("=" * 55)
print(f"  ✔ Total titles analyzed       : {len(df):,}")
print(f"  ✔ Movies                      : {(df['type']=='Movie').sum():,}")
print(f"  ✔ TV Shows                    : {(df['type']=='TV Show').sum():,}")
print(f"  ✔ Most common rating          : {df['rating'].mode()[0]}")
print(f"  ✔ Top country                 : {df['country'].dropna().mode()[0]}")
print(f"  ✔ Top genre                   : {genre_counts.index[0]}")
print(f"  ✔ Content peak year (release) : {release.idxmax()}")
print("=" * 55)
print("  Task 3 Complete. Ready for GitHub upload!")
print("=" * 55)
