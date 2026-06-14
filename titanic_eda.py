# ============================================================
# Titanic Dataset — Exploratory Data Analysis (EDA)
# CodeAlpha Data Analytics Internship — Task 2
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────
# Using seaborn's built-in Titanic dataset (no download needed)
df = sns.load_dataset('titanic')

print("=" * 55)
print("         TITANIC DATASET — EDA REPORT")
print("=" * 55)

# ─────────────────────────────────────────
# 2. BASIC OVERVIEW
# ─────────────────────────────────────────
print("\n📌 SHAPE:", df.shape)
print("\n📌 COLUMNS:\n", df.columns.tolist())
print("\n📌 DATA TYPES:\n", df.dtypes)
print("\n📌 FIRST 5 ROWS:\n", df.head())

# ─────────────────────────────────────────
# 3. MISSING VALUES
# ─────────────────────────────────────────
print("\n📌 MISSING VALUES:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing, 'Percentage (%)': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0])

# ─────────────────────────────────────────
# 4. BASIC STATISTICS
# ─────────────────────────────────────────
print("\n📌 DESCRIPTIVE STATISTICS:\n", df.describe())

# ─────────────────────────────────────────
# 5. SURVIVAL RATE OVERVIEW
# ─────────────────────────────────────────
survival_rate = df['survived'].value_counts(normalize=True) * 100
print("\n📌 SURVIVAL RATE:")
print(f"  Survived : {survival_rate[1]:.1f}%")
print(f"  Died     : {survival_rate[0]:.1f}%")

# ─────────────────────────────────────────
# 6. VISUALIZATIONS
# ─────────────────────────────────────────
sns.set_style("darkgrid")
sns.set_palette("Set2")

fig, axes = plt.subplots(3, 2, figsize=(14, 16))
fig.suptitle("Titanic EDA — Key Insights", fontsize=18, fontweight='bold', y=1.01)

# --- Plot 1: Survival Count ---
sns.countplot(x='survived', data=df, ax=axes[0, 0], palette=['#e74c3c', '#2ecc71'])
axes[0, 0].set_title('Survival Count')
axes[0, 0].set_xticklabels(['Did Not Survive', 'Survived'])
axes[0, 0].set_xlabel('')

# --- Plot 2: Survival by Gender ---
sns.countplot(x='sex', hue='survived', data=df, ax=axes[0, 1], palette=['#e74c3c', '#2ecc71'])
axes[0, 1].set_title('Survival by Gender')
axes[0, 1].legend(['Did Not Survive', 'Survived'])

# --- Plot 3: Survival by Passenger Class ---
sns.countplot(x='pclass', hue='survived', data=df, ax=axes[1, 0], palette=['#e74c3c', '#2ecc71'])
axes[1, 0].set_title('Survival by Passenger Class')
axes[1, 0].set_xlabel('Class (1 = First, 3 = Third)')
axes[1, 0].legend(['Did Not Survive', 'Survived'])

# --- Plot 4: Age Distribution ---
axes[1, 1].set_title('Age Distribution by Survival')
df[df['survived'] == 0]['age'].dropna().plot(kind='hist', alpha=0.6, color='#e74c3c',
                                              label='Did Not Survive', ax=axes[1, 1], bins=30)
df[df['survived'] == 1]['age'].dropna().plot(kind='hist', alpha=0.6, color='#2ecc71',
                                              label='Survived', ax=axes[1, 1], bins=30)
axes[1, 1].legend()
axes[1, 1].set_xlabel('Age')

# --- Plot 5: Fare Distribution ---
sns.boxplot(x='pclass', y='fare', data=df, ax=axes[2, 0], palette='Set2')
axes[2, 0].set_title('Fare Distribution by Class')
axes[2, 0].set_xlabel('Passenger Class')

# --- Plot 6: Correlation Heatmap ---
numeric_df = df[['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']].dropna()
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[2, 1], linewidths=0.5)
axes[2, 1].set_title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('titanic_eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Plots saved as 'titanic_eda_plots.png'")

# ─────────────────────────────────────────
# 7. KEY FINDINGS SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("               KEY FINDINGS")
print("=" * 55)

female_survival = df[df['sex'] == 'female']['survived'].mean() * 100
male_survival = df[df['sex'] == 'male']['survived'].mean() * 100
class1_survival = df[df['pclass'] == 1]['survived'].mean() * 100
class3_survival = df[df['pclass'] == 3]['survived'].mean() * 100
avg_age_survived = df[df['survived'] == 1]['age'].mean()
avg_age_died = df[df['survived'] == 0]['age'].mean()

print(f"\n  ✔ Overall survival rate       : {survival_rate[1]:.1f}%")
print(f"  ✔ Female survival rate        : {female_survival:.1f}%")
print(f"  ✔ Male survival rate          : {male_survival:.1f}%")
print(f"  ✔ 1st Class survival rate     : {class1_survival:.1f}%")
print(f"  ✔ 3rd Class survival rate     : {class3_survival:.1f}%")
print(f"  ✔ Avg age of survivors        : {avg_age_survived:.1f} years")
print(f"  ✔ Avg age of non-survivors    : {avg_age_died:.1f} years")
print(f"\n  📝 'Age' has ~20% missing values — handle before modeling.")
print(f"  📝 Fare and Pclass are negatively correlated (expected).")
print(f"  📝 Gender is the strongest survival predictor.")
print("\n" + "=" * 55)
print("  EDA Complete. Ready for GitHub upload!")
print("=" * 55)
