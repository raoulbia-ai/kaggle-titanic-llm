from config import FEATURE_IMPORTANCE
import numpy as np
import pandas as pd

HISTORICAL_CONTEXT = """
Inconsistent Protocols: The "women and children first" protocol was applied inconsistently. First Officer William Murdoch 
interpreted it to mean women and children first, while Second Officer Charles Lightoller adhered strictly to that, 
leading to disparities in survival rates.
Panic and Delays: Panic erupted among passengers, especially in third class, where many were unaware of the severity of 
the situation and struggled to reach the lifeboats due to flooding and confusion.
Musicians' Role: none survived."""

# System message for AI
SYSTEM_MESSAGE = """You are an assistant that predicts Titanic passenger survival. 
    Analyze ALL the given information carefully, including age, gender, class, economic tier, family situation, 
    cabin position, cultural background, and the overall survival advantage score. 
    Provide your reasoning, considering how each factor typically influenced survival chances. 
    Ensure your final conclusion ('Survived' or 'Did not survive') is consistent with your reasoning. 
    Your analysis should be thorough and logical, reflecting the complex interplay of factors that influenced 
    survival on the Titanic."""

def generate_prompt(passenger, is_train=True):
    narrative = create_narrative(passenger)
    analysis = create_analytical_breakdown(passenger)
    comparison = create_comparative_analysis(passenger) if is_train else ""
    uncertainty = identify_uncertainty_factors(passenger)

    prompt = f"""Predict the survival of the following Titanic passenger using step-by-step reasoning:

Passenger Information:
{narrative}

Analytical Breakdown:
{analysis}

Historical Context:
{HISTORICAL_CONTEXT}

{comparison}

{uncertainty}

Please provide your prediction by following these steps:

1. Analyze the key factors affecting this passenger's survival chances.
2. Compare the passenger's profile to the historical survival rates.
3. Consider any uncertainty factors and their potential impact.
4. Weigh the evidence for and against survival.
5. Make a final prediction and explain your confidence level.

When responding make sure to provide your step-by-step reasoning, concluding with your prediction and confidence level (low, medium, or high)."""

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

def create_assistant_message(passenger):
    survived = "survived" if passenger['Survived'] == 1 else "did not survive"
    reasons = []
    
    if passenger['Sex'] == 'female':
        reasons.append("being female generally increased survival chances")
    else:
        reasons.append("being male generally reduced survival chances")
    
    if passenger['Pclass'] == 1:
        reasons.append("first-class passengers had better survival rates")
    elif passenger['Pclass'] == 3:
        reasons.append("third-class passengers had lower survival rates")
    
    if passenger['AgeBin'] in ['Child', 'Teenager']:
        reasons.append("children were often prioritized for rescue")
    
    if passenger['FareBin'] in ['High', 'Very High']:
        reasons.append("passengers with expensive tickets may have had better access to lifeboats")
    
    if passenger['FamilySize'] > 1:
        reasons.append(f"traveling with {passenger['FamilySize'] - 1} family member(s) could have influenced survival")
    
    reasoning = " and ".join(reasons)
    return f"Based on the information provided, {passenger['Title']} {passenger['LastName']} {survived}. " \
           f"Reasoning: {reasoning.capitalize() if reasoning else 'Multiple factors influenced survival rates, including class, gender, age, and location on the ship.'}"
