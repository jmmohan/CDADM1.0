# cdadm_evaluation.py
from cdadm_framework import cdadm_framework
from utils import load_data, load_model

def evaluate_cdadm(dataset, model):
    """
    Evaluate CDADM mechanism on adversarial data.
    
    Args:
        dataset (tuple): Tuple of (X_test, y_test).
        model (object): Trained ML model.
        
        
    Returns:
        dict: Results of CDADM evaluation.
    """
    X_test, y_test = dataset
    return cdadm_framework(X_test, model)
