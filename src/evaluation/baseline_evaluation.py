# baseline_evaluation.py
from utils import load_data

def evaluate_baseline(model, dataset):
    """
    Evaluate baseline performance of a model on clean data.
    
    
    Args:
        model (object): Trained ML model.
        dataset (tuple): Tuple of (X_test, y_test).
        
    Returns:
        float: Accuracy on clean data.
    """
    X_test, y_test = dataset
    accuracy = model.score(X_test, y_test)
    print(f"Baseline Accuracy: {accuracy}")
    return accuracy