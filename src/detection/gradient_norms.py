# gradient_norms.py
import numpy as np

def compute_gradient_norms(input_data, model):
    """
    Compute gradient norms for adversarial detection.
    
    Args:
        input_data (np.ndarray): Input samples.
        model (object): Trained ML model.
        
    Returns:
        np.ndarray: Gradient norms for each input.
    """
    gradients = np.abs(np.gradient(model.predict(input_data)))
    return np.linalg.norm(gradients, axis=-1)
