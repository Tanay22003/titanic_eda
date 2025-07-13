# Titanic Dataset - Exploratory Data Analysis
# Author: [Your Name]
# Date: [Current Date]

# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set plot style
sns.set(style="whitegrid")
plt.style.use('ggplot')

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Dataset info
print("\nDataset Info:")
print(df.info())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# =========================
# Data Cleaning
# =========================

# Fill missing 'Age' with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing 'Embarked' with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' due to high % of missing values
df.drop(columns='Cabin', inplace=True)

# Drop 'Ticket' (not useful for EDA)
df.drop(columns='Ticket', inplace=True)

# =========================
# Univariate Analysis
# =========================

# Countplot - Survival Distribution
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Countplot - Sex Distribution
sns.countplot(x='Sex', data=df)
plt.title('Sex Distribution')
plt.show()

# Countplot - Passenger Class
sns.countplot(x='Pclass', data=df)
plt.title('Passenger Class Distribution')
plt.show()

# Histogram - Age Distribution
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Histogram - Fare Distribution
sns.histplot(df['Fare'], bins=40, kde=True)
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()

# =========================
# Bivariate Analysis
# =========================

# Survival rate by Sex
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Sex')
plt.show()

# Survival rate by Pclass
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Passenger Class')
plt.show()

# Boxplot - Age vs Survived
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Age Distribution by Survival')
plt.show()

# Countplot - Embarked vs Survived
sns.countplot(x='Embarked', hue='Survived', data=df)
plt.title('Survival Count by Embarkation Port')
plt.show()

# Barplot - SibSp vs Survived
sns.barplot(x='SibSp', y='Survived', data=df)
plt.title('Survival Rate by SibSp (Siblings/Spouses Aboard)')
plt.show()

# Barplot - Parch vs Survived
sns.barplot(x='Parch', y='Survived', data=df)
plt.title('Survival Rate by Parch (Parents/Children Aboard)')
plt.show()

# =========================
# Outlier Detection
# =========================

# Boxplot - Fare
sns.boxplot(x=df['Fare'])
plt.title('Boxplot of Fare')
plt.show()

# =========================
# Correlation Analysis
# =========================

# Select numerical features
numerical_features = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
corr_matrix = df[numerical_features].corr()

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# =========================
# Multivariate Analysis (Optional)
# =========================

# Pairplot
sns.pairplot(df[numerical_features], hue='Survived')
plt.suptitle('Pairplot of Numerical Features by Survival', y=1.02)
plt.show()

# =========================
# Summary of Insights
# =========================

print("\n📝 Summary of Key Insights:")
print("- Females had a much higher survival rate than males.")
print("- 1st class passengers had better chances of survival than 2nd and 3rd class.")
print("- Younger passengers (especially children) were more likely to survive.")
print("- Passengers with high fare tickets were more likely to survive.")
print("- Passengers who embarked from Cherbourg (C) had higher survival rates.")
print("- Those traveling alone or with very large families had lower survival chances.")
