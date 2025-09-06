# Part 1: Setup and Data Loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings to keep the output clean
import warnings
warnings.filterwarnings('ignore')

# Set a style for the plots for better aesthetics
sns.set(style="whitegrid")

# Load the dataset
# Dataset source: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
# We'll assume the dataset is in a CSV file named 'bank-additional-full.csv'
# It's important to specify the separator as it's not a standard comma
try:
    df = pd.read_csv('bank-additional-full.csv', sep=';')
except FileNotFoundError:
    print("Error: The 'bank-additional-full.csv' file was not found. Please download the dataset from the UCI Machine Learning Repository and place it in the same directory.")
    print("Link: https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip")
    exit()

print("Dataset loaded successfully! Let's get started with EDA. 🕵️‍♀️")

# Part 2: Initial Data Inspection
print("\n## Initial Data Inspection")

# Display the first 5 rows of the dataframe to get a sense of the data
print("First 5 rows of the dataset:")
print(df.head())

# Display the dimensions (rows and columns) of the dataset
print(f"\nDataset has {df.shape[0]} rows and {df.shape[1]} columns.")

# Get a concise summary of the dataframe, including data types and non-null values
print("\nDataFrame Info:")
df.info()

# Get descriptive statistics for numerical columns
print("\nDescriptive Statistics for Numerical Columns:")
print(df.describe())

# Get descriptive statistics for categorical/object columns
print("\nDescriptive Statistics for Categorical Columns:")
print(df.describe(include='object'))

# Check for duplicate rows
print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")
if df.duplicated().sum() > 0:
    df.drop_duplicates(inplace=True)
    print("Duplicate rows have been removed.")

# Part 3: Data Cleaning and Preprocessing
print("\n## Data Cleaning and Preprocessing")

# The 'poutcome' column has a category 'nonexistent' which we will map to a more intuitive name.
df['poutcome'] = df['poutcome'].replace('nonexistent', 'no_previous_campaign')

# Let's map the target variable 'y' (which indicates subscription) to a more readable format
# 'yes' becomes 1, and 'no' becomes 0.
df['y'] = df['y'].map({'yes': 1, 'no': 0})
print("\nTarget variable 'y' has been mapped to 1 (yes) and 0 (no).")

# Part 4: Univariate Analysis
print("\n## Univariate Analysis")
print("Visualizing the distribution of key variables.")

# Distribution of the target variable 'y'
plt.figure(figsize=(8, 6))
sns.countplot(x='y', data=df)
plt.title('Distribution of the Target Variable (Subscribed to Term Deposit)')
plt.xticks([0, 1], ['No', 'Yes'])
plt.xlabel('Subscription Status')
plt.ylabel('Count')
plt.show()

# Distribution of 'age'
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], kde=True, bins=30)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Distribution of 'job'
plt.figure(figsize=(12, 8))
sns.countplot(y='job', data=df, order=df['job'].value_counts().index)
plt.title('Distribution of Job Categories')
plt.xlabel('Count')
plt.ylabel('Job')
plt.show()


# Part 5: Bivariate and Multivariate Analysis
print("\n## Bivariate and Multivariate Analysis")
print("Exploring relationships between variables.")

# Relationship between 'age' and 'y' (subscription)
plt.figure(figsize=(10, 6))
sns.boxplot(x='y', y='age', data=df)
plt.title('Age Distribution by Subscription Status')
plt.xticks([0, 1], ['No', 'Yes'])
plt.xlabel('Subscription Status')
plt.ylabel('Age')
plt.show()

# Relationship between 'job' and 'y'
plt.figure(figsize=(12, 8))
sns.countplot(y='job', hue='y', data=df, order=df['job'].value_counts().index)
plt.title('Subscription Status by Job Category')
plt.xlabel('Count')
plt.ylabel('Job')
plt.legend(title='Subscribed', labels=['No', 'Yes'])
plt.show()


# Correlation heatmap for numerical features
plt.figure(figsize=(12, 10))
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
correlation_matrix = df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# Let's look at the relationship between 'duration' and 'y' (subscription)
# Note: 'duration' is a very important feature but should not be used in a predictive model.
# It represents the last contact duration, which is known after the outcome is decided.
plt.figure(figsize=(10, 6))
sns.boxplot(x='y', y='duration', data=df)
plt.title('Call Duration by Subscription Status')
plt.xticks([0, 1], ['No', 'Yes'])
plt.xlabel('Subscription Status')
plt.ylabel('Duration (in seconds)')
plt.show()

# Part 6: Insights and Summary
print("\n## Key Insights from EDA")
print("Based on the analysis, here are some key findings:")
print(f"1. A total of {df['y'].value_counts()[1]} clients subscribed to the term deposit, which is "
      f"about {df['y'].mean() * 100:.2f}% of the dataset.")
print("2. The average age of subscribed clients is slightly higher than non-subscribed clients.")
print("3. Job categories like 'student' and 'retired' have a higher subscription rate relative to their total numbers.")
print("4. There's a strong positive correlation between 'euribor3m' and 'emp.var.rate', which suggests "
      "they are related economic indicators. This can be important for feature selection.")
print("5. The 'duration' of the last contact is a highly predictive feature for the outcome. "
      "Clients with longer calls were much more likely to subscribe. However, this is for analysis only and "
      "should not be used in a predictive model as it's a look-ahead variable.")
