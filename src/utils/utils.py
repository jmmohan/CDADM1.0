# utils.py
import pickle

def load_data(filepath):
    """
    Load dataset from a pickle file.
    
    Args:
        filepath (str): Path to the dataset file.
        
    Returns:
        tuple: Loaded dataset (X, y).
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def load_model(filepath):
    """
    Load a pre-trained model from file.
    
    Args:
        filepath (str): Path to the model file.
        
    Returns:
        object: Loaded model.
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)
