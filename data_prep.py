import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import *

def prepare_data(file_path, is_train=True):
    df = pd.read_csv(file_path)
    
    # Extract LastName
    df['LastName'] = df['Name'].str.split(',').str[0]
    
    # Feature engineering
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms', 'Mme'], ['Miss', 'Miss', 'Mrs'])
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Deck'] = df['Cabin'].str[0].fillna('U')
    df['NameLength'] = df['Name'].apply(len)
    
    # Handle missing data
    df['Age'] = df.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
    df['Fare'] = df.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Create FareBin as a categorical type
    df['FareBin'] = pd.qcut(df['Fare'], FARE_BINS, labels=FARE_LABELS).astype(str)
    
    df['AgeBin'] = pd.cut(df['Age'], bins=AGE_BINS, labels=AGE_LABELS).astype(str)
    
    # Store original values before scaling
    df['Age_Original'] = df['Age']
    df['Fare_Original'] = df['Fare']
    
    # Interaction features
    df['Pclass_Age'] = df['Pclass'] * df['Age']
    df['Sex_Fare'] = df['Sex'].map({'male': 0, 'female': 1}) * df['Fare']
    
    # Social status feature
    df['SocialStatus'] = df['Pclass'].astype(str) + '_' + df['Title']
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_features = ['Age', 'Fare', 'FamilySize', 'NameLength', 'Pclass_Age', 'Sex_Fare']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    if is_train:
        # Create family survival rate feature for training data
        df['FamilySurvivalRate'] = df.groupby('LastName')['Survived'].transform('mean')
    else:
        # For test data, we'll use a placeholder value
        df['FamilySurvivalRate'] = -1
    
    if not is_train:
        # For test data, we'll use a placeholder value for Survived
        df['Survived'] = -1

    return df