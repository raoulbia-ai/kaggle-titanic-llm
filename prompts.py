from config import HISTORICAL_CONTEXT, FEATURE_IMPORTANCE
import numpy as np
import pandas as pd

def generate_prompt(passenger, is_train=True):
    narrative = create_narrative(passenger)
    analysis = create_analytical_breakdown(passenger)
    
    # Simplified and clear prompt for the model
    prompt = f"""You are an AI assistant trained to analyze Titanic passenger data and predict survival outcomes.
                 {narrative}
              """

    return prompt

def create_narrative(passenger):
    age = int(passenger['Age_Original'])
    age_desc = "elderly" if age > 65 else "middle-aged" if age > 30 else "young"
    class_desc = ["first", "second", "third"][passenger['Pclass'] - 1]
    family_size = max(1, round(passenger['FamilySize']))  # Ensure minimum of 1
    family_desc = "alone" if family_size == 1 else f"with {family_size - 1} family member(s)"
    
    # Map embarkation ports
    embarkation_ports = {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}
    embarkation = embarkation_ports.get(passenger['Embarked'], 'an unknown port')
    
    narrative = f"{passenger['Title']} {passenger['LastName']}, a {age}-year-old {passenger['Sex']} passenger, "
    narrative += f"embarked on the Titanic's maiden voyage in {class_desc}-class. "
    narrative += f"Traveling {family_desc}, they boarded at {embarkation} "
    narrative += f"with a {passenger['FareBin'].lower()} fare. "
    
    if pd.notna(passenger['Cabin']) and passenger['Cabin'] != 'U':
        narrative += f"Their cabin was {passenger['Cabin']}."
    else:
        narrative += "Their specific cabin location is unknown."
    
    return narrative

def create_analytical_breakdown(passenger):
    analysis = f"{passenger['Title']} {passenger['LastName']}'s survival chances are influenced by several factors: "
    
    if passenger['Sex'] == 'male':
        analysis += "Being male significantly reduces survival probability due to the 'women and children first' protocol. "
    else:
        analysis += "Being female increases survival chances due to the 'women and children first' protocol. "
    
    analysis += f"Their {passenger['AgeBin'].lower()} age of {int(passenger['Age_Original'])} "
    if passenger['AgeBin'] in ['Child', 'Teenager']:
        analysis += "is advantageous. "
    elif passenger['AgeBin'] == 'Elderly':
        analysis += "may be disadvantageous. "
    else:
        analysis += "is neutral. "
    
    analysis += f"Traveling in {['first', 'second', 'third'][passenger['Pclass'] - 1]}-class "
    if passenger['Pclass'] == 1:
        analysis += "provides better access to lifeboats. "
    elif passenger['Pclass'] == 3:
        analysis += "limits access to lifeboats. "
    else:
        analysis += "offers moderate lifeboat access. "
    
    family_size = max(1, round(passenger['FamilySize']))  # Ensure minimum of 1
    if family_size == 1:
        analysis += "Traveling alone may affect decision-making during the crisis. "
    else:
        analysis += f"Traveling with {family_size - 1} family member(s) could influence evacuation choices. "
    
    # Interpret social status
    class_desc = ["first", "second", "third"][passenger['Pclass'] - 1]
    analysis += f"Their status as a {passenger['Title']} in {class_desc}-class may affect treatment during evacuation."
    
    return analysis

def create_comparative_analysis(passenger):
    # These statistics are derived from historical analyses of the Titanic passenger list
    survival_rates = {
        ('male', 1): 0.34,  # First-class males had a survival rate of approximately 34%
        ('male', 2): 0.15,  # Second-class males had a survival rate of approximately 15%
        ('male', 3): 0.13,  # Third-class males had a survival rate of approximately 13%
        ('female', 1): 0.97,  # First-class females had a survival rate of approximately 97%
        ('female', 2): 0.86,  # Second-class females had a survival rate of approximately 86%
        ('female', 3): 0.47   # Third-class females had a survival rate of approximately 47%
    }

    overall_rate = survival_rates.get((passenger['Sex'], passenger['Pclass']), 0.3)
    age_group_rate = overall_rate * 1.1 if passenger['AgeBin'] in ['Child', 'Teenager'] else overall_rate * 0.9

    return f"""Comparative Analysis: 
    Among {passenger['Sex']} passengers in {['first', 'second', 'third'][passenger['Pclass'] - 1]} class, 
    the historical survival rate was approximately {overall_rate:.0%}. 
    For passengers of similar age ({passenger['AgeBin'].lower()}), the estimated survival rate is about {age_group_rate:.0%}."""

def identify_uncertainty_factors(passenger):
    factors = []
    if np.isclose(passenger['Age'], passenger['Age_Original']):
        factors.append("The passenger's exact age was unknown and has been estimated.")
    if np.isclose(passenger['Fare'], passenger['Fare_Original']):
        factors.append("The passenger's exact fare was unknown and has been estimated.")
    if passenger['Cabin'] == 'U':
        factors.append("The passenger's cabin location is unknown.")
    
    if factors:
        return "Uncertainty factors: " + " ".join(factors)
    else:
        return "No significant uncertainty factors identified."
