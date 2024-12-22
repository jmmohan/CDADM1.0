# tabular_model.py
from sklearn.ensemble import RandomForestClassifier

def get_tabular_model():
    """
    Load a RandomForest model for tabular data classification.
    
    Returns:
        model: RandomForestClassifier instance.
    """
    return RandomForestClassifier(n_estimators=100)

