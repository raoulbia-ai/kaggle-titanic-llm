import pandas as pd
import numpy as np
from sklearn.utils import resample

# Load the data
data = pd.read_csv('data/train.csv')

def selective_oversample(df, condition, n_samples):
    subset = df[condition]
    if len(subset) == 0:
        return pd.DataFrame()
    return resample(subset, replace=True, n_samples=n_samples, random_state=42)

# Original counts
original_survivors = len(data[data['Survived'] == 1])
original_non_survivors = len(data[data['Survived'] == 0])

# Oversample based on historical factors
balanced_data = pd.DataFrame()

# 1. Women and children in first and second class (high survival rate)
balanced_data = pd.concat([balanced_data, selective_oversample(
    data, (data['Survived'] == 1) & (data['Sex'] == 'female') & (data['Pclass'].isin([1, 2])), 
    n_samples=int(original_non_survivors * 0.2))])

# 2. Men in first class (moderate survival rate)
balanced_data = pd.concat([balanced_data, selective_oversample(
    data, (data['Survived'] == 1) & (data['Sex'] == 'male') & (data['Pclass'] == 1), 
    n_samples=int(original_non_survivors * 0.1))])

# 3. Women and children in third class (lower survival rate)
balanced_data = pd.concat([balanced_data, selective_oversample(
    data, (data['Survived'] == 1) & ((data['Sex'] == 'female') | (data['Age'] < 18)) & (data['Pclass'] == 3), 
    n_samples=int(original_non_survivors * 0.1))])

# 4. Crew members (some survived)
balanced_data = pd.concat([balanced_data, selective_oversample(
    data, (data['Survived'] == 1) & (data['Fare'] == 0), 
    n_samples=int(original_non_survivors * 0.05))])

# 5. Add remaining survivors to reach balance
remaining_survivors_needed = original_non_survivors - len(balanced_data)
balanced_data = pd.concat([balanced_data, selective_oversample(
    data, (data['Survived'] == 1) & (~data.index.isin(balanced_data.index)), 
    n_samples=remaining_survivors_needed)])

# Add all original non-survivors
balanced_data = pd.concat([balanced_data, data[data['Survived'] == 0]])

# Shuffle the dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Display the new class distribution
print(balanced_data['Survived'].value_counts())

# Save the balanced dataset
balanced_data.to_csv('data/balanced_train.csv', index=False)

print("Historically balanced dataset saved as 'historically_balanced_train.csv'")

# Additional statistics
print("\nClass distribution in balanced dataset:")
print(balanced_data['Pclass'].value_counts(normalize=True))

print("\nGender distribution in balanced dataset:")
print(balanced_data['Sex'].value_counts(normalize=True))

print("\nAverage age in balanced dataset:", balanced_data['Age'].mean())