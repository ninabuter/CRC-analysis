import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "data/colorectal_cancer_dataset.csv"
df = pd.read_csv(file_path)

### 1. BASIC INFO AND DATA CLEANING ###
print("Dataset Overview:\n", df.info())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing Values:\n", missing_values[missing_values > 0])

# Summary statistics
print("\nBasic Statistics:\n", df.describe())

# Check unique values in categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
print("\nCategorical Columns:\n", categorical_cols)
for col in categorical_cols:
    print(f"\nUnique values in {col}:\n {df[col].nunique()}")


### 2. COUNTRY-WISE CANCER INCIDENCE AND MORTALITY ###
plt.figure(figsize = (12, 8))
top_countries = df["Country"].value_counts()[:10]
print("\nTop 10 countries:\n", top_countries)
sns.barplot(x=top_countries.index, y=top_countries.values, palette="coolwarm")
plt.title("Top 10 Countries with Highest Colorectal Cancer Cases")
plt.ylabel("Number of Cases")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize = (12, 8))
mortality_rates = df.groupby("Country")["Mortality_Rate_per_100K"].mean().sort_values(ascending=False)[:10]
print("\nMortality Rates:\n", mortality_rates)
sns.barplot(x=mortality_rates.index, y=mortality_rates.values, palette="Reds_r")
plt.title("Top 10 Countries with Highest Mortality Rates")
plt.ylabel("Average Mortality Rate per 100K")
plt.xticks(rotation=45)
plt.show()


### 3. AGE AND GENDER IMPACT ON SURVIVAL ###
plt.figure(figsize=(12, 6))
sns.histplot(df[df["Survival_5_years"] == "Yes"]["Age"], bins=30, kde=True, color="green", label="Survived")
sns.histplot(df[df["Survival_5_years"] == "No"]["Age"], bins=30, kde=True, color="red", label="Not Survived")
plt.title("Age Distribution of Survived vs. Not Survived Patients")
plt.xlabel("Age")
plt.legend()
plt.show()

# Gender impact
plt.figure(figsize=(8, 5))
sns.countplot(x="Gender", hue="Survival_5_years", data=df, palette="viridis")
plt.title("Survival Rate by Gender")
plt.ylabel("Count")
plt.show()


### 4. CANCER STAGE DISTRIBUTION ###
plt.figure(figsize=(10, 5))
sns.countplot(x="Cancer_Stage", data=df, palette="coolwarm", order=["Localized", "Regional", "Metastatic"])
plt.title("Cancer Stage Distribution")
plt.ylabel("Number of Patients")
plt.show()

# Survival by cancer stage
plt.figure(figsize=(10, 5))
sns.countplot(x="Cancer_Stage", hue="Survival_5_years", data=df, palette="pastel")
plt.title("Survival Rate by Cancer Stage")
plt.show()


### 5. LIFESTYLE RISK FACTOR ANALYSIS ###
risk_factors = ["Smoking_History", "Alcohol_Consumption", "Obesity_BMI", "Diet_Risk", "Physical_Activity", "Diabetes"]
for factor in risk_factors:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=factor, hue="Survival_5_years", data=df, palette="magma")
    plt.title(f"Survival Impact by {factor}")
    plt.show()


### 6. HEALTHCARE COST VARIATION ###
plt.figure(figsize=(12, 6))
sns.boxplot(x="Economic_Classification", y="Healthcare_Costs", data=df, palette="coolwarm")
plt.title("Healthcare Costs by Economic Classification")
plt.ylabel("Cost in $")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x="Cancer_Stage", y="Healthcare_Costs", data=df, palette="mako")
plt.title("Healthcare Costs by Cancer Stage")
plt.ylabel("Cost in $")
plt.show()


### 7. TREATMENT EFFECTIVENESS ANALYSIS ###
plt.figure(figsize=(12, 6))
sns.countplot(x="Treatment_Type", hue="Survival_5_years", data=df, palette="Set2")
plt.title("Survival Rate by Treatment Type")
plt.xticks(rotation=45)
plt.show()

# Cost vs. Treatment
plt.figure(figsize=(12, 6))
sns.boxplot(x="Treatment_Type", y="Healthcare_Costs", data=df, palette="rocket")
plt.title("Healthcare Costs for Different Treatments")
plt.xticks(rotation=45)
plt.ylabel("Cost in $")
plt.show()

